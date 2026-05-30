from dotenv import load_dotenv
load_dotenv()

#-------key components-----------
from src.document_loader import load_documents, split_documents
from src.vector_store import create_vector_store_with_embeddings, load_documents_into_vector_store
from src.ui import start_authentication

#----evaluations-------
from evals.ragas_eval import run_evaluation, run_agent_evaluation


def main():
    # create Vector Store with Embeddings (_ ist die Python-Konvention für "wird nicht gebraucht)
    vector_store = create_vector_store_with_embeddings()

    # Nur laden und splitten, wenn die DB noch leer ist
    if vector_store.get(limit=1)["ids"]:
        print("✅ Vector Store bereits befüllt — Laden und Splitting übersprungen")
        start_authentication(vector_store)
    else:
        docs = load_documents()
        chunks = split_documents(docs)
        load_documents_into_vector_store(vector_store, chunks)

        #run_evaluation(vector_store)
        #run_agent_evaluation(vector_store, get_agent_response)
        #start_agent(vector_store)
        start_authentication(vector_store)


if __name__ == "__main__":
    main()
#add info about store (amound of embeddings, ids etc.) -> good to konw logging
