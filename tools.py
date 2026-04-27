import requests
import html2text
from bs4 import BeautifulSoup
from langchain_core.tools import tool

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
MAX_URL_READS = 10


# HTML zu Text Konverter 
_html_to_text = html2text.HTML2Text()
_html_to_text.ignore_links = True    # Links weglassen, nur Textinhalt relevant
_html_to_text.ignore_images = True   # Bilder weglassen
_html_to_text.body_width = 0         # Keine automatischen Zeilenumbrüche


def _fetch_page_text(url: str, max_chars: int = 3000) -> str:
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
    """Durchsucht das Zürcher Steuerbuch (zh.ch) nach aktuellen kantonalen Steuervorschriften.
    Verwende dieses Tool wenn die lokalen Dokumente keine ausreichende Antwort liefern
    oder wenn explizit nach Züricher kantonalen Regelungen gefragt wird.
    Gibt Titel, ZStB-Nummer und Textinhalt der relevantesten Einträge zurück."""

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

    entries = search_data.get("data", [])
    if not entries:
        return f"Keine Steuerbuch-Einträge für '{query}' gefunden."

    total_results = search_data.get("numberOfResults", len(entries))

    # 2: Seiteninhalte laden
    # Jeder Eintrag enthält ein "link"-Feld mit dem relativen Pfad zur Steuerbuch-Seite.
    collected = []

    for entry in entries[:MAX_URL_READS]:
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

    # 3: Ergebnisse zusammenführen und zurückgeben
    # Trennzeichen zwischen Einträgen erleichtert dem LLM die Strukturerkennung.
    header = (
        f"Steuerbuch-Suche: '{query}' – "
        f"{total_results} Treffer gefunden, {len(collected)} Einträge geladen.\n\n"
    )
    return header + "\n\n---\n\n".join(collected)


# Lokaler Test 
# Wird nur ausgeführt wenn python tools.py gestartet wird
# Beim Import durch main.py wird dieser Block ignoriert.
if __name__ == "__main__":
    test_query = "Miete"
    print(f"Teste Tool mit Query: '{test_query}'\n")
    print("=" * 60)

    # .invoke() ist der LangChain-Aufruf für Tools (statt direktem Funktionsaufruf)
    result = search_steuerbuch.invoke({"query": test_query})
    print(result)
