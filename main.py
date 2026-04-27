from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
#from langchain.agents import create_tool_calling_agent, AgentExecutor

#----RAGAS-----
from evals.ragas_eval import run_evaluation, generate_testset

#----Tools-----
from tools import search_steuerbuch

# Für einfache RAG-Abfragen (Evaluation)
llm = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    #Antwort auf 200 tokens beschränken
    max_tokens=200,
    temperature=0
)

# ---> Noah: Für den Agent (mehr Tokens für Reasoning + Tool-Calls)
agent_llm = init_chat_model(
    "gpt-4o",
    model_provider="openai",
)
file_path = "ressources/steuergesetz.pdf"

def load_documents(file_path):
    #load PDFs
    loader: PyPDFLoader = PyPDFLoader(file_path, mode="page")
    docs: list[Document] = loader.load() #page_content, metadata
    return docs

def split_documents(docs: list[Document]):
    #create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "] #Sätze werden bestmöglich bei Zeilenumbrüchen abgeschnitten und nicht mitten im Satz
    )
    #split text into chunks
    chunks = text_splitter.split_documents(docs)
    #print(chunks) test
    return chunks

def create_vector_store_with_embeddings():
    # create embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #create vector_store
    vector_store: Chroma = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
            #cosin space
            collection_metadata = {"hnsw:space": "cosine"}
        )
    return vector_store, embeddings

def make_ids(chunks: list[Document]) -> list[str]:
    return [f"{c.metadata.get('source', 'unknown')}_{i}" for i, c in enumerate(chunks)]

def load_documents_into_vector_store(vector_store: Chroma, chunks: list[Document]):
    ids = make_ids(chunks)
    if vector_store.get(ids[:1])["ids"]:
        print("✅ Already in store")
    else:
        # load chunks with ids into vector store, chunks gets automatically embedded
        vector_store.add_documents(documents=chunks, ids=ids)
        print(f"{len(ids)} new chunks added to store")

def ask_chatbot(query: str, vector_store: Chroma):
    # implement retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    #search
    results_with_scores = vector_store.similarity_search_with_score(query, k=5)
    query_score = results_with_scores[0][-1]
    #search vector store
    retrieve_data = retriever.invoke(query)
    #save metadata from received chunk for output
    metadata = retrieve_data[0].metadata
    # Contexts für RAGAS extrahieren
    contexts = [doc.page_content for doc in retrieve_data]
    # Prompt / chain begins
    #Wenn die Antwort nicht im Kontext steht, sag: "Da bin ich mir nicht sicher...
    template = """Du bist ein Assistent für Steuerberatung. Nutze NUR den folgenden Kontext,
    um die Frage zu beantworten".
    Kontext:
    {context}
    Frage: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    question = query
    answer = rag_chain.invoke(question)
    print(f"KI-Antwort:\n{answer}\n {query_score:.2f} | {metadata.get('source', 'n/a')} | page {metadata.get('page', 'n/a')}")
    return answer, contexts

def start_bot(vector_store: Chroma):
    loop = True
    while loop:
        print("-------- Ask me anything ---------")
        query = input()
        if query in ["exit", "quit"]:
            print("Thank you and goodbye!")
            loop = False
        else:
            ask_chatbot(query, vector_store)

def main():
    #load ressources
    docs = load_documents(file_path)
    #generate chunks from document
    split_documents(docs)
    #create vector_store with embedding model
    vector_store, embeddings = create_vector_store_with_embeddings()
    #load chunks into vector_store
    load_documents_into_vector_store(vector_store, docs)
    #generate testset
    generate_testset(docs, llm, embeddings, testset_size=10)
    #run evaluation
    #run_evaluation(vector_store, ask_chatbot)
    #start bot with retrieving information
    start_bot(vector_store)

if __name__ == "__main__":
    main()
#add info about store (amound of embeddings, ids etc.) -> good to konw logging