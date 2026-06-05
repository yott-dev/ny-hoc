from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.retriever import retrieve_local_context


def test_retrieve_local_context_empty_database():
    """Ensures that an empty vector store is handled gracefully without hitting the reranker."""
    fake_vector_store = MagicMock()
    # Simulate database returning absolutely nothing for all multi-queries
    fake_vector_store.similarity_search.return_value = []

    with patch("src.retriever.generate_queries", return_value=["Alt 1", "Alt 2"]):
        context_str, raw_docs = retrieve_local_context("Abzug Pendlerkosten", fake_vector_store)

    assert context_str == "Keine relevanten Dokumente in der lokalen Datenbank gefunden."
    assert raw_docs == []


def test_retrieve_local_context_underflow_handling():
    """Validates stability when unique documents found are fewer than TOP_K_RERANKED."""
    fake_vector_store = MagicMock()

    # Only a single unique document is returned by the database
    doc = Document(page_content="Kinderabzug Regelung", metadata={"source": "stb.pdf", "page": 12})
    fake_vector_store.similarity_search.return_value = [doc]

    fake_model = MagicMock()
    fake_model.predict.return_value = [0.99]  # Exactly one score required for a single document

    with patch("src.retriever.generate_queries", return_value=[]), \
            patch("src.retriever._get_reranker", return_value=fake_model):
        context_str, raw_docs = retrieve_local_context("Kinderabzug", fake_vector_store)

    assert len(raw_docs) == 1
    assert raw_docs[0].page_content == "Kinderabzug Regelung"
    assert "[Quelle: stb.pdf, Seite 12]" in context_str


def test_retrieve_local_context_metadata_hierarchy():
    """Verifies that h2 takes precedence over h1, and missing page fields are caught."""
    fake_vector_store = MagicMock()

    # Use distinct, non-overlapping names to prevent accidental substring matches
    doc_1 = Document(
        page_content="Content A",
        metadata={"source": "docA.pdf", "page": 1, "h1": "Alpha-Titel", "h2": "Beta-Unterkapitel"}
    )
    doc_2 = Document(
        page_content="Content B",
        metadata={"source": "docB.pdf", "h1": "Gamma-Titel"}
    )

    # Simulate 4 loop iterations (1 main query + 3 generated alternative queries)
    fake_vector_store.similarity_search.side_effect = [[doc_1], [doc_2], [], []]

    fake_model = MagicMock()
    fake_model.predict.return_value = [0.80, 0.70]

    with patch("src.retriever.generate_queries", return_value=["Alt 1", "Alt 2", "Alt 3"]), \
            patch("src.retriever._get_reranker", return_value=fake_model):
        context_str, _ = retrieve_local_context("Test Query", fake_vector_store)

    # 1. Verify h2 takes precedence over h1 by making sure Alpha-Titel was discarded
    assert "Beta-Unterkapitel" in context_str
    assert "Alpha-Titel" not in context_str

    # 2. Verify fallback to h1 when h2 is completely absent
    assert "Quelle: docB.pdf, Gamma-Titel" in context_str
    assert "Seite" not in context_str.split("\n\n")[1]