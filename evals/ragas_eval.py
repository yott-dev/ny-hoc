from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
import os
import pandas as pd
import asyncio
from ragas import experiment, Dataset
from ragas.metrics.collections import AnswerAccuracy, Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

#ask_chatbot
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

# Wird nur von run_evaluation verwendet
def ask_chatbot(query: str, vector_store: Chroma, llm):
    # implement retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    #search
    #results_with_scores = vector_store.similarity_search_with_score(query, k=5) delete?
    #query_score = results_with_scores[0][-1] delete?
    #search vector store
    retrieve_data = retriever.invoke(query)
    #save metadata from received chunk for output
    #metadata = retrieve_data[0].metadata delete?
    # Contexts für RAGAS extrahieren
    contexts = [doc.page_content for doc in retrieve_data]
    # Prompt / chain begins
    # Wenn die Antwort nicht im Kontext steht, sag: "Da bin ich mir nicht sicher...
    template = """Du bist ein hilfsbereiter Assistent für Steuerberatung. Nutze NUR den folgenden Kontext,
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
    return answer, contexts

#generates testset with ragas (not implemented due to rags issues)
def generate_testset(docs, ragas_llm, ragas_embeddings, testset_size=10):
    csv_path = "evals/datasets/data.csv"

    if os.path.exists(csv_path):
        print("✅ Testset bereits vorhanden, wird geladen...")
        return pd.read_csv(csv_path)

    print("🔄 Generiere neues Testset...")

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings
    )

    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=testset_size,
    )

    df = testset.to_pandas()
    df.to_csv(csv_path, index=False)
    print("✅ Testset gespeichert")

    return df

def load_dataset():
    dataset = Dataset.load(
        name="data",
        backend="local/csv",
        root_dir="./evals"
    )
    return dataset

def run_evaluation(vector_store):
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o", client=client)
    langchain_llm = init_chat_model("gpt-4o", model_provider="openai")
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-large",
        client=client,
        interface="modern"
    )

    # Alle Metriken initialisieren
    answer_accuracy = AnswerAccuracy(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    dataset = load_dataset()

    @experiment()
    async def run_experiment(row):
        answer, contexts = ask_chatbot(row["user_input"], vector_store, langchain_llm)

        # Alle Metriken parallel berechnen
        accuracy, faith, relevancy, precision, recall = await asyncio.gather(
            answer_accuracy.ascore(user_input=row["user_input"], response=answer, reference=row["reference"]),
            faithfulness.ascore(user_input=row["user_input"], response=answer, retrieved_contexts=contexts),
            answer_relevancy.ascore(user_input=row["user_input"], response=answer),
            context_precision.ascore(user_input=row["user_input"], retrieved_contexts=contexts, reference=row["reference"]),
            context_recall.ascore(user_input=row["user_input"], retrieved_contexts=contexts, reference=row["reference"]),
        )

        return {
            **row,
            "response": answer,
            "answer_accuracy": accuracy.value,
            "faithfulness": faith.value,
            "answer_relevancy": relevancy.value,
            "context_precision": precision.value,
            "context_recall": recall.value,
        }

    results = asyncio.run(run_experiment.arun(dataset, name="steuerbot_v1"))
    print(results.to_pandas().to_string())


def run_agent_evaluation(vector_store, agent_fn):
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o", client=client)
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-large",
        client=client,
        interface="modern"
    )

    # Alle Metriken initialisieren
    answer_accuracy = AnswerAccuracy(llm=llm)
    faithfulness = Faithfulness(llm=llm)

    dataset = load_dataset()

    @experiment()
    async def run_experiment(row):
        result, rag_context = agent_fn(row["user_input"], vector_store, [])
        answer = result["messages"][-1].content

        # RAG-Chunks aus dem formatierten String rausschneiden
        rag_chunks = [block.strip() for block in rag_context.split("\n\n") if block.strip()]

        # Tool-Texte aus ToolMessages extrahieren
        tool_texts = [msg.content for msg in result["messages"] if isinstance(msg, ToolMessage)]

        contexts = rag_chunks + tool_texts

        # Alle Metriken parallel berechnen
        accuracy, faith = await asyncio.gather(
            answer_accuracy.ascore(user_input=row["user_input"], response=answer, reference=row["reference"]),
            faithfulness.ascore(user_input=row["user_input"], response=answer, retrieved_contexts=contexts),
        )

        return {
            **row,
            "response": answer,
            "answer_accuracy": accuracy.value,
            "faithfulness": faith.value,
        }

    results = asyncio.run(run_experiment.arun(dataset, name="steuerbot_agent_v1"))
    print(results.to_pandas().to_string())





