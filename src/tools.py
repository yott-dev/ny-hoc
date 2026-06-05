import requests
import html2text
from typing import List, Dict, Any, Tuple, Set, Optional, Callable

from langchain_core.tools import tool, BaseTool
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from sentence_transformers import CrossEncoder
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

# Initialize lightweight model for query expansion/generation
_query_llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)

# Lazy loading container for the heavy CrossEncoder model
_reranker: Optional[CrossEncoder] = None

TOP_K_RERANKED = 6
BASE_URL = "https://www.zh.ch"

# Internal API endpoint of the Zurich Tax Book (Adobe Experience Manager CMS via JCR selectors)
SEARCH_API: str = (
        BASE_URL
        + "/de/steuern-finanzen/steuern/treuhaender/steuerbuch"
        + "/jcr:content/main/flexdatasearch.zhweb-flexdata.zhweb-cache.json"
)

MAX_API_RESULTS: int = 30
MAX_URL_READS: int = 3

# Configure HTML-to-Text rendering pipeline
_html_to_text = html2text.HTML2Text()
_html_to_text.ignore_links = True
_html_to_text.ignore_images = True
_html_to_text.body_width = 0

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/msmarco-MiniLM-L12-en-de-v1")
    return _reranker


def generate_queries(question: str, n: int = 3) -> List[str]:
    """Generates alternative tax-specific search queries using domain terminology."""
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
    queries: List[str] = [
        q.strip()
        for q in str(response.content).strip().split("\n")
        if q.strip()
    ]
    return queries[:n]


def _fetch_page_text(url: str, max_chars: int = 10000) -> str:
    """Fetches server-side rendered tax law articles and extracts the core content container."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract structural content container to eliminate boilerplate global navigation elements
    content = soup.select_one("div.mod-smaller-headings") or soup.find("main")
    target_html: str = str(content) if content else response.text

    return _html_to_text.handle(target_html)[:max_chars].strip()


@tool
def search_steuerbuch(query: str) -> str:
    """Search the official Zurich Tax Book (ZStB) for cantonal tax regulations,
    rates, deductions, and legal deadlines. Pass 1-2 precise keywords only.
    """
    params: Dict[str, Any] = {
        "keyword": query,
        "page": 1,
        "order": "ascending",
        "orderBy": "recordfield-flexdataschemainputfzstb-nummer",
    }

    try:
        api_response = requests.get(SEARCH_API, params=params, timeout=10)
        api_response.raise_for_status()
        search_data: Dict[str, Any] = api_response.json()
    except Exception as e:
        return f"API request failed: {e}"

    entries: List[Dict[str, Any]] = search_data.get("data", [])[:MAX_API_RESULTS]
    if not entries:
        return f"No tax book entries found for '{query}'."

    total_results = search_data.get("numberOfResults", len(entries))

    # Fuzzy string matching to prioritize exact hits in article titles
    for entry in entries:
        title = entry.get("recordfield-flexdataschemainputftitel", "")
        entry["_score"] = fuzz.partial_ratio(query.lower(), title.lower())

    entries = sorted(entries, key=lambda e: e["_score"], reverse=True)[:MAX_URL_READS]
    collected: List[str] = []

    for entry in entries:
        relative_link = entry.get("link", "")
        title = entry.get("recordfield-flexdataschemainputftitel", "No Title")
        zstb_nr = entry.get("recordfield-flexdataschemainputfzstb-nummer", "?")

        if not relative_link:
            continue

        full_url = BASE_URL + relative_link

        try:
            page_text = _fetch_page_text(full_url)
            collected.append(
                f"[ZStB Nr. {zstb_nr} – {title}]\n"
                f"Source: {full_url}\n\n"
                f"{page_text}"
            )
        except Exception as e:
            collected.append(
                f"[ZStB Nr. {zstb_nr} – {title}] Error loading page ({full_url}): {e}"
            )

    if not collected:
        return "Search results found, but contents could not be retrieved."

    header = (
        f"Tax Book Search: '{query}' – "
        f"{total_results} matches found, {len(collected)} entries compiled.\n\n"
    )
    return header + "\n\n---\n\n".join(collected)


def create_local_search_tool(vector_store: Chroma) -> BaseTool:
    """Factory closure to initialize the local RAG tool with a persistent vector store instance."""

    @tool
    def search_local_documents(query: str) -> str:
        """Searches local federal tax laws and circulars. Use this tool prior to web tools."""
        alt_queries = generate_queries(query, n=3)
        all_queries = [query] + alt_queries
        print(f"🔍 Multi-Query: {all_queries}")

        seen_ids: Set[str] = set()
        all_results: List[Any] = []

        # Multi-query execution across the vector database
        for q in all_queries:
            for doc in vector_store.similarity_search(q, k=5):
                # Deterministic composite key identification for deduplication
                doc_id = (
                    f"{doc.metadata.get('source')}_"
                    f"{doc.metadata.get('page')}_"
                    f"{doc.page_content[:80]}"
                )
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)

        if not all_results:
            return "No relevant documents found in the local vector database."

        print(f"📄 Found {len(all_results)} unique chunks from {len(all_queries)} queries")

        # Reranking stage using the localized cross-encoder model
        scores = _get_reranker().predict([(query, doc.page_content) for doc in all_results])
        ranked = sorted(zip(scores, all_results), key=lambda x: x[0], reverse=True)
        all_results = [doc for _, doc in ranked[:TOP_K_RERANKED]]
        print(f"🏆 Retained {len(all_results)} chunks after cross-encoder reranking")

        parts: List[str] = []
        for doc in all_results:
            meta = doc.metadata
            section = meta.get("h2") or meta.get("h1") or ""
            section_str = f", {section}" if section else ""
            page_str = f", Page {meta.get('page')}" if meta.get("page") is not None else ""
            parts.append(
                f"[Source: {meta.get('source', '?')}{page_str}{section_str}]\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    return search_local_documents


if __name__ == "__main__":
    import requests as _req

    test_query = "Miete"
    print(f"\nDebug execution for query: '{test_query}'")

    raw = _req.get(SEARCH_API, params={"keyword": test_query, "page": 1}, timeout=10).json()
    debug_entries = raw.get("data", [])[:MAX_API_RESULTS]
    print(f"Step 1 – API: {raw.get('numberOfResults')} hits, {len(debug_entries)} payload items fetched.")

    for e in debug_entries:
        e["_score"] = fuzz.partial_ratio(test_query.lower(), e.get("recordfield-flexdataschemainputftitel", "").lower())
    top = sorted(debug_entries, key=lambda entry_item: entry_item["_score"], reverse=True)[:MAX_URL_READS]

    print(f"Step 2 – Scoring Top {MAX_URL_READS}:")
    for e in top:
        print(
            f"  [{e['_score']:.1f}] {e.get('recordfield-flexdataschemainputfzstb-nummer', '?')} – {e.get('recordfield-flexdataschemainputftitel', '?')}")

    print(f"Step 3 – Content Extraction:")
    for e in top:
        test_url = BASE_URL + e.get("link", "")
        text_content = _fetch_page_text(test_url)
        print(f"  {test_url} → {len(text_content)} characters extracted.")