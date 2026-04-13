from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document #
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from openai.types import vector_store


def main():
    # https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader

    llm = init_chat_model("gpt-4o-mini")
    file_path = "ressources/steuergesetz.pdf"



    loader: PyPDFLoader = PyPDFLoader(file_path, mode="page")
    docs: list[Document] = loader.load() #page_content, metadata
    #split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    #split chunks into text
    texts = text_splitter.split_documents(docs)
    #create embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store: Chroma = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"  # Where to save data locally, remove if not necessary
    )

    #load embedded chunks into vector store
    vector_store.add_documents(documents=texts)

    #InMemory Vector Store(old)
    """vector_store = InMemoryVectorStore(embedding=embeddings)"""

    #implement retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    #retriever.retrieve(query="Hey")

    #chain begins
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
    frage = "Lottogewinn versteuern?"
    antwort = rag_chain.invoke(frage)
    print(f"KI-Antwort:\n{antwort}")



if __name__ == "__main__":
    main()
