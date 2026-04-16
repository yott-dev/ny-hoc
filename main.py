from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document #
from langchain_openai import OpenAIEmbeddings
#from langchain_core.vectorstores import InMemoryVectorStore (old)
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

llm = init_chat_model("gpt-4o-mini")
file_path = "ressources/steuergesetz.pdf"

def load_documents():
    #load page
    loader: PyPDFLoader = PyPDFLoader(file_path, mode="page")
    docs: list[Document] = loader.load() #page_content, metadata
    #metadata = docs[0].metadata
    #print(metadata)
    return docs

def split_documents(docs: list[Document]):
    #create text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
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
            persist_directory="./chroma_langchain_db"
        )
    return vector_store

def load_documents_into_vector_store(vector_store: Chroma, chunks: list[Document]):
    #load chunks into vector store, chunks gets automatically embedded
    vector_store.add_documents(documents=chunks)


def ask_chatbot(query: str, vector_store: Chroma):
    # implement retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    retrieve_data = retriever.invoke(query)
    metadata = retrieve_data[0].metadata
    # Prompt / chain begins
    template = """Du bist ein Assistent für Steuerberatung. Nutze NUR den folgenden Kontext,
    um die Frage zu beantworten. Wenn die Antwort nicht im Kontext steht, sag: "Dazu habe ich keine Info".
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
    print(f"KI-Antwort:\n{answer}\n Found in {metadata.get('source', 'n/a')} on page {metadata.get('page', 'n/a')}")

def main():
    #load ressources
    docs = load_documents()
    #generate chunks from document
    split_documents(docs)
    #create vector_store with embedding model
    vector_store = create_vector_store_with_embeddings()
    #load chunks into vector_store
    load_documents_into_vector_store(vector_store, docs)
    #ask chatbot a question
    loop = True
    while loop:
        print("-------- Ask me anything ---------")
        print("question:")
        query = input()
        if query in ["exit", "quit"]:
            print("Thank you and goodbye!")
            loop = False
        else:
            ask_chatbot(query, vector_store)


if __name__ == "__main__":
    main()
