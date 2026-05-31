from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_query_llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)

_reranker: CrossEncoder | None = None
TOP_K_RERANKED = 6


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/msmarco-MiniLM-L12-en-de-v1")
    return _reranker


def generate_queries(question: str, n: int = 3) -> list[str]:
    """Generiert n alternative Suchanfragen in Steuer-Fachsprache via gpt-4o-mini."""
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
    """RAG-First Pipeline: Multi-Query → ChromaDB → Reranking.

    Gibt (formatierter Kontext-String für den Agent, Top-Dokumente für RAGAS) zurück.
    Der String wird direkt in die HumanMessage eingefügt.
    Die Dokument-Liste ermöglicht saubere context_precision/recall Messung in der Evaluation.
    """
    alt_queries = generate_queries(query, n=3)
    all_queries = [query] + alt_queries
    print(f"🔍 Multi-Query: {all_queries}")

    seen_ids: set[str] = set()
    all_results: list[Document] = []

    for q in all_queries:
        for doc in vector_store.similarity_search(q, k=5):
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

    print(f"📄 {len(all_results)} einzigartige Chunks (aus {len(all_queries)} Queries)")

    scores = _get_reranker().predict([(query, doc.page_content) for doc in all_results])
    ranked = sorted(zip(scores, all_results), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:TOP_K_RERANKED]]
    print(f"🏆 Nach Reranking: {len(top_docs)} Chunks behalten")

    parts = []
    for doc in top_docs:
        meta = doc.metadata
        section = meta.get("h2") or meta.get("h1") or ""
        section_str = f", {section}" if section else ""
        page_str = f", Seite {meta.get('page')}" if meta.get("page") is not None else ""
        parts.append(
            f"[Quelle: {meta.get('source', '?')}{page_str}{section_str}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts), top_docs
