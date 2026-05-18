from dotenv import load_dotenv
load_dotenv()

from main import create_vector_store_with_embeddings

vector_store, _ = create_vector_store_with_embeddings()
results = vector_store.get(limit=10, include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    print("METADATEN:", meta)
    print("INHALT:", doc[:1000])
    print("---")

print(f"\nTotal Chunks im Store: {vector_store._collection.count()}")
