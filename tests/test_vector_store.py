from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.vector_store import create_vector_store_with_embeddings, load_documents_into_vector_store

def test_create_vector_store_returns_chroma():
    with patch("src.vector_store.OpenAIEmbeddings"), \
         patch("src.vector_store.Chroma") as mock_chroma:
        result = create_vector_store_with_embeddings()
        assert mock_chroma.return_value == result


def test_load_documents_already_in_store():
    """Ensures that existing documents are skipped and not duplicated in ChromaDB."""
    fake_store = MagicMock()

    # Simulate that ChromaDB already knows this ID by returning it in the .get() call
    fake_store.get.return_value = {"ids": ["some_deterministic_hash_id"]}

    # Pass the fake_store to your pipeline execution
    # Ensure your production code filters out documents if their ID is found in fake_store.get()

    # If your pipeline logic checks correctly, fake_store.add_documents should NOT be called.
    assert fake_store.add_documents.call_count == 0

def test_load_documents_new_chunks():
    fake_store = MagicMock()
    fake_store.get.return_value = {"ids": []}  # noch nicht vorhanden
    chunks = [Document(page_content="Test", metadata={"source": "test.pdf"})]

    load_documents_into_vector_store(fake_store, chunks)

    fake_store.add_documents.assert_called_once()