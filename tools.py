import requests
import html2text

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from sentence_transformers import CrossEncoder

# informationen können von einer webseite als HTML oder XML gescraped werden
from bs4 import BeautifulSoup

# misst wie ähnlich zwei Texte sind, auch wenn sie nicht exakt übereinstimmen (günstiger als embedding)
from rapidfuzz import fuzz

# Kleines Modell nur für Query-Generierung
_query_llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)

# Modell wird erst beim ersten Aufruf geladen
_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/msmarco-MiniLM-L12-en-de-v1")
    return _reranker

TOP_K_RERANKED = 6  # Nur die 6 relevantesten Chunks an den Agent weitergeben


def generate_queries(question: str, n: int = 3) -> list[str]:
    """Generiert n alternative Suchanfragen für eine gegebene Frage.

    Warum: Nutzer formulieren Fragen umgangssprachlich ("zuhause arbeiten"),
    aber Steuerdokumente verwenden Fachbegriffe ("Homeoffice-Abzug", "Arbeitszimmer").
    Mehrere Varianten erhöhen die Chance, relevante Chunks zu finden.
    """
    prompt = (
        f"Du bist ein Schweizer Steuerrecht-Experte. "
        f"Formuliere {n} alternative Suchanfragen für folgende Frage. "
        f"Jede Anfrage soll 1-3 präzise Steuerfachbegriffe enthalten, "
        f"die in Steuergesetzen oder ESTV-Kreisschreiben vorkommen würden. "
        f"Füge KEINE Jahreszahlen hinzu. "
        f"Gib NUR die Anfragen zurück, eine pro Zeile, ohne Nummerierung oder Erklärung.\n\n"
        f"Frage: {question}"
    )
    response = _query_llm.invoke(prompt)
    # Jede Zeile ist eine Anfrage — leere Zeilen werden gefiltert
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return queries[:n]



# ── Konstanten
BASE_URL = "https://www.zh.ch"

# Interner API-Endpoint des Zürcher Steuerbuchs.
# Gefunden via Chrome DevTools → Network → Fetch/XHR während einer Suche
# Der Endpoint verwendet JCR-Selektoren (Adobe Experience Manager CMS).
SEARCH_API = (
    BASE_URL
    + "/de/steuern-finanzen/steuern/treuhaender/steuerbuch"
    + "/jcr:content/main/flexdatasearch.zhweb-flexdata.zhweb-cache.json"
)

# Maximale Anzahl Steuerbuch-Seiten die pro Tool-Aufruf geladen werden (verhindert endlos loops)
MAX_API_RESULTS = 30   # Wie viele Einträge die API zurückgeben soll (nur JSON)
MAX_URL_READS   = 3    # Wie viele Seiten davon wirklich geladen werden (langsam)



# HTML zu Text Konverter 
_html_to_text = html2text.HTML2Text()
_html_to_text.ignore_links = True    # Links weglassen, nur Textinhalt relevant
_html_to_text.ignore_images = True   # Bilder weglassen
_html_to_text.body_width = 0         # Keine automatischen Zeilenumbrüche


def _fetch_page_text(url: str, max_chars: int = 10000) -> str:
    """Lädt eine Steuerbuch-Seite und gibt nur den Dokumentinhalt als Text zurück.

    Die Seiten sind server-side gerendert → kein JavaScript / kein Headless-Browser nötig.
    BeautifulSoup extrahiert zuerst den Inhaltsbereich (div.mod-smaller-headings),
    damit Navigation, Header und Footer nicht im Output landen.
    max_chars begrenzt die Ausgabe damit das LLM-Kontextfenster nicht überflutet wird.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Inhaltscontainer der zh.ch Steuerbuch-Seiten (Adobe Experience Manager CMS).
    # Enthält den reinen Dokumenttext ohne globale Navigation und Footer.
    # Fallback auf <main> falls der spezifische Container nicht gefunden wird.
    content = soup.select_one("div.mod-smaller-headings") or soup.find("main")
    target_html = str(content) if content else response.text

    return _html_to_text.handle(target_html)[:max_chars].strip()


@tool
def search_steuerbuch(query: str) -> str:
    """Durchsucht das Zürcher Steuerbuch (zh.ch), die verbindliche Rechtsquelle für kantonales Steuerrecht.

    WANN AUFRUFEN:
    - Bei Fragen zu kantonalen Zürich spezifischen Regelungen
    - Bei Fragen zu Steuertarifen, Abzugslimiten oder gesetzlichen Fristen
    - Wenn lokale Dokumente keine eindeutige Antwort liefern
    Nicht nötig bei: allgemeinen Definitionen die lokal vollständig abgedeckt sind.

    QUERY-REGEL (wichtig): Übergib immer nur 1-2 präzise Schlüsselwörter, keine ganzen Sätze.
    Beispiele: "Homeoffice", "Miete Abzug", "Quellensteuer", "Erbschaft"

    THEMEN IM ZÜRCHER STEUERBUCH:
    Steuerpflicht, Steuerbefreiung, Einkommenssteuer, Vermögenssteuer, Abzüge,
    Steuertarife, Quellensteuer, Grundstückgewinnsteuer, Erbschaftssteuer,
    Schenkungssteuer, Handänderungssteuer, Verrechnungssteuer,
    juristische Personen, Steuerverfahren, Einsprache, Rekurs

    Gibt Titel, ZStB-Nummer, Quellenlink und Volltext der relevantesten Einträge zurück."""

    # 1: Steuerbuch-API abfragen
    # Die API gibt eine JSON-Liste mit passenden Steuerbuch-Einträgen zurück.
    # Sortierung nach ZStB-Nummer sorgt für eine konsistente, nachvollziehbare Reihenfolge.
    params = {
        "keyword": query,
        "page": 1,                                                    # Erste Ergebnisseite
        "order": "ascending",
        "orderBy": "recordfield-flexdataschemainputfzstb-nummer",     # Sortierung nach ZStB-Nr.
    }

    try:
        api_response = requests.get(SEARCH_API, params=params, timeout=10)
        api_response.raise_for_status()
        search_data = api_response.json()
    except Exception as e:
        return f"API-Abfrage fehlgeschlagen: {e}"

    entries = search_data.get("data", [])[:MAX_API_RESULTS]
    if not entries:
        return f"Keine Steuerbuch-Einträge für '{query}' gefunden."

    total_results = search_data.get("numberOfResults", len(entries))

    # 2: Titel-Scoring mit rapidfuzz — relevanteste Einträge zuerst laden
    # partial_ratio prüft ob die Query als Teilstring im Titel vorkommt
    for entry in entries:
        title = entry.get("recordfield-flexdataschemainputftitel", "")
        entry["_score"] = fuzz.partial_ratio(query.lower(), title.lower())

    entries = sorted(entries, key=lambda e: e["_score"], reverse=True)[:MAX_URL_READS]

    # 3: Seiteninhalte der relevantesten Einträge laden
    # Jeder Eintrag enthält ein "link"-Feld mit dem relativen Pfad zur Steuerbuch-Seite.
    collected = []

    for entry in entries:
        relative_link = entry.get("link", "")
        title = entry.get("recordfield-flexdataschemainputftitel", "Kein Titel")
        zstb_nr = entry.get("recordfield-flexdataschemainputfzstb-nummer", "?")

        if not relative_link:
            continue

        full_url = BASE_URL + relative_link

        try:
            # HTML-Seite laden und in Klartext umwandeln
            page_text = _fetch_page_text(full_url)
            collected.append(
                f"[ZStB Nr. {zstb_nr} – {title}]\n"
                f"Quelle: {full_url}\n\n"
                f"{page_text}"
            )
        except Exception as e:
            # Einzelne fehlerhafte Seiten überspringen, nicht den ganzen Aufruf abbrechen
            collected.append(
                f"[ZStB Nr. {zstb_nr} – {title}] Fehler beim Laden ({full_url}): {e}"
            )

    if not collected:
        return "Suchergebnisse gefunden, aber keine Seiten konnten geladen werden."

    # 4: Ergebnisse zusammenführen und zurückgeben
    # Trennzeichen zwischen Einträgen erleichtert dem LLM die Strukturerkennung.
    header = (
        f"Steuerbuch-Suche: '{query}' – "
        f"{total_results} Treffer gefunden, {len(collected)} Einträge geladen.\n\n"
    )
    return header + "\n\n---\n\n".join(collected)


def create_local_search_tool(vector_store: Chroma):
    # Factory: Erstellt das RAG-Tool mit dem vector_store als Closure.
    # So bleibt alle Tool-Logik in tools.py, main.py liefert nur den vector_store.
    @tool
    def search_local_documents(query: str) -> str:
        """Durchsucht die lokale Datenbank mit Schweizer Steuergesetzen und ESTV-Kreisschreiben.
        Verwende dieses Tool zuerst, bevor du das Online-Steuerbuch durchsuchst."""

        # Schritt 1: 3 alternative Suchanfragen generieren
        # Zusammen mit der Originalfrage haben wir 4 Queries total
        alt_queries = generate_queries(query, n=3)
        all_queries = [query] + alt_queries
        print(f"🔍 Multi-Query: {all_queries}")

        # Schritt 2: Für jede Query die ChromaDB durchsuchen
        # seen_ids verhindert, dass derselbe Chunk mehrfach im Kontext landet
        seen_ids: set[str] = set()
        all_results = []

        for q in all_queries:
            for doc in vector_store.similarity_search(q, k=5):
                # Eindeutige ID pro Chunk: Dateiname + Seite + erster Textinhalt
                # (page_content[:80] reicht, da zwei identische Chunks denselben Anfang haben)
                doc_id = (
                    f"{doc.metadata.get('source')}_"
                    f"{doc.metadata.get('page')}_"
                    f"{doc.page_content[:80]}"
                )
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)

        if not all_results:
            return "Keine relevanten Dokumente in der lokalen Datenbank gefunden."

        print(f"📄 {len(all_results)} einzigartige Chunks gefunden (aus {len(all_queries)} Queries)")

        # Schritt 3: Reranking — Cross-Encoder bewertet (Originalfrage, Chunk)-Paare
        # Vorteil gegenüber Embedding-Similarity: Modell sieht Query und Chunk zusammen
        # → erkennt semantische Relevanz präziser, filtert irrelevante Chunks heraus
        scores = _get_reranker().predict([(query, doc.page_content) for doc in all_results])
        ranked = sorted(zip(scores, all_results), key=lambda x: x[0], reverse=True)
        all_results = [doc for _, doc in ranked[:TOP_K_RERANKED]]
        print(f"🏆 Nach Reranking: {len(all_results)} Chunks behalten")

        # Schritt 4: Ergebnisse formatieren — gleich wie vorher
        # Jeden Treffer mit Quellenangabe formatieren damit das LLM die Herkunft kennt
        # section = nächstgelegene Überschrift aus dem MarkdownHeaderTextSplitter (h2 bevorzugt)
        parts = []
        for doc in all_results:
            meta = doc.metadata
            section = meta.get("h2") or meta.get("h1") or ""
            section_str = f", {section}" if section else ""
            page_str = f", Seite {meta.get('page')}" if meta.get("page") is not None else ""
            parts.append(
                f"[Quelle: {meta.get('source', '?')}{page_str}{section_str}]\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    return search_local_documents


# Lokaler Test — wird nur ausgeführt wenn `uv run python tools.py` gestartet wird.
if __name__ == "__main__":
    import requests as _req
    test_query = "Miete"
    print(f"\nQuery: '{test_query}'")

    # Schritt 1: API
    raw = _req.get(SEARCH_API, params={"keyword": test_query, "page": 1}, timeout=10).json()
    entries = raw.get("data", [])[:MAX_API_RESULTS]
    print(f"Schritt 1 – API: {raw.get('numberOfResults')} Treffer, {len(entries)} geladen")

    # Schritt 2: Scoring
    for e in entries:
        e["_score"] = fuzz.partial_ratio(test_query.lower(), e.get("recordfield-flexdataschemainputftitel","").lower())
    top = sorted(entries, key=lambda e: e["_score"], reverse=True)[:MAX_URL_READS]
    print(f"Schritt 2 – Scoring Top {MAX_URL_READS}:")
    for e in top:
        print(f"  [{e['_score']}] {e.get('recordfield-flexdataschemainputfzstb-nummer','?')} – {e.get('recordfield-flexdataschemainputftitel','?')}")

    # Schritt 3: Seite laden
    print(f"Schritt 3 – Seiteninhalte:")
    for e in top:
        url = BASE_URL + e.get("link", "")
        text = _fetch_page_text(url)
        print(f"  {url} → {len(text)} Zeichen")

    # .invoke() ist der LangChain-Aufruf für Tools (statt direktem Funktionsaufruf)
    # result = search_steuerbuch.invoke({"query": test_query})
    # print(result)
