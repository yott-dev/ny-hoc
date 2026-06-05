import os
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from src.config import TOP_K_RERANKED, MULTI_QUERY_COUNT, SIMILARITY_SEARCH_K


# Lazy-loaded singletons for memory efficiency
_query_llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Thread-safe style lazy initializer for the CrossEncoder reranking model."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/msmarco-MiniLM-L12-en-de-v1")
    return _reranker


def generate_queries(question: str, n: int = MULTI_QUERY_COUNT) -> list[str]:
    """Generates alternative search queries optimized for Swiss tax domain terminology."""
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
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return queries[:n]


def retrieve_local_context(query: str, vector_store: Chroma) -> tuple[str, list[Document]]:
    """
    Executes a multi-stage retrieval pipeline consisting of Multi-Query, Vector Search, and Reranking.

    Returns a formatted string block of context for the prompt injection and a raw list of top-scoring
    documents dedicated for downstream RAGAS evaluation.
    """
    alt_queries = generate_queries(query, n=MULTI_QUERY_COUNT)
    all_queries = [query] + alt_queries
    print(f"🔍 Multi-Query Expansion: {all_queries}")

    seen_ids: set[str] = set()
    all_results: list[Document] = []

    # Two-pass deduplication layout during cross-query vector fetching
    for q in all_queries:
        for doc in vector_store.similarity_search(q, k=SIMILARITY_SEARCH_K):
            doc_id = (
                f"{doc.metadata.get('source')}_"
                f"{doc.metadata.get('page')}_"
                f"{doc.page_content[:80]}"
            )
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append(doc)

    if not all_results:
        return "Keine relevanten Dokumente in der lokalen Datenbank gefunden.", []

    print(f"📄 Found {len(all_results)} unique chunks across {len(all_queries)} queries.")

    # Compute semantic alignment scores via cross-encoder
    scores = _get_reranker().predict([(query, doc.page_content) for doc in all_results])
    ranked = sorted(zip(scores, all_results), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:TOP_K_RERANKED]]

    print(f"🏆 Reranking complete: Keeping top {len(top_docs)} chunks.")

    # Format the context blocks with legal metadata citations for the agent's system prompt
    parts = []
    for doc in top_docs:
        meta = doc.metadata
        section = meta.get("h2") or meta.get("h1") or ""
        section_str = f", {section}" if section else ""
        page_str = f", Seite {meta.get('page')}" if meta.get('page') is not None else ""

        parts.append(
            f"[Quelle: {meta.get('source', '?')}{page_str}{section_str}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts), top_docs