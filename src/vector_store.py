from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.utils_parsing import make_ids


def create_vector_store_with_embeddings() -> Chroma:
    # create embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store: Chroma = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
            collection_metadata = {"hnsw:space": "cosine"}
        )
    return vector_store

def load_documents_into_vector_store(vector_store: Chroma, chunks: list[Document]):
    ids = make_ids(chunks)
    if vector_store.get(ids[:1])["ids"]:
        print("All chunks already in store")
    else:
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
            print(f"  Batch {i // batch_size + 1}: {len(batch_chunks)} Chunks gespeichert")
        print(f"✅ {len(ids)} Chunks total in Store geladen")