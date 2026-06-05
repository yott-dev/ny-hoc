from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.utils_parsing import make_ids
from src.config import BATCH_SIZE

load_dotenv()


def create_vector_store_with_embeddings() -> Chroma:
    """
    Initializes a Chroma vector store instance using OpenAI's high-dimensional embeddings.

    Configures the under-the-hood vector space to use cosine similarity distances.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store


def load_documents_into_vector_store(vector_store: Chroma, chunks: list[Document]) -> None:
    """
    Validates chunk content uniqueness and uploads missing documents in batches.
    """
    ids = make_ids(chunks)
    if not ids:
        print("⚠️ No chunks provided for vector store ingestion.")
        return

    # Fetch existing document tracking records to filter out overlap duplicates
    existing_records = vector_store.get(ids=ids)
    existing_ids = set(existing_records.get("ids", []))

    # Isolate only the unique chunks that do not exist inside our active collection
    new_chunks = [c for c, chunk_id in zip(chunks, ids) if chunk_id not in existing_ids]
    new_ids = [chunk_id for chunk_id in ids if chunk_id not in existing_ids]

    if not new_chunks:
        print("ℹ️ Ingestion skipped: All provided chunks already exist within the storage collection.")
        return

    # Process and save the novel chunks using windowed batch processing splits
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch_chunks = new_chunks[i:i + BATCH_SIZE]
        batch_ids = new_ids[i:i + BATCH_SIZE]

        vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
        print(f"  Batch {i // BATCH_SIZE + 1}: Saved {len(batch_chunks)} new chunks.")

    print(f"✅ Ingestion complete: Loaded {len(new_ids)} new chunks into the vector store.")