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

def generate_testset(docs: list[Document], ragas_llm, ragas_embeddings, testset_size: int = 10, force_regenerate: bool = False):
    csv_path = "evals/datasets/data.csv"

    if os.path.exists(csv_path) and not force_regenerate:
        print("✅ Testset bereits vorhanden, wird geladen...")
        return pd.read_csv(csv_path)

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        language="german"
    )

    testset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)

    df = testset.to_pandas()
    df.to_csv(csv_path, index=False)
    print("✅ Testset gespeichert")

    return df  # immer df zurückgeben, beide Pfade konsistent

def load_dataset():
    dataset = Dataset.load(
        name="data",
        backend="local/csv",
        root_dir="./evals"
    )
    return dataset

def run_evaluation(vector_store, chatbot_fn):
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
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    dataset = load_dataset()

    @experiment()
    async def run_experiment(row):
        answer, contexts = chatbot_fn(row["user_input"], vector_store)

        # Alle Metriken parallel berechnen
        accuracy, faith, relevancy, precision, recall = await asyncio.gather(
            answer_accuracy.ascore(user_input=row["user_input"], response=answer, reference=row["reference"]),
            faithfulness.ascore(user_input=row["user_input"], response=answer, retrieved_contexts=contexts),
            answer_relevancy.ascore(user_input=row["user_input"], response=answer),
            context_precision.ascore(user_input=row["user_input"], retrieved_contexts=contexts,
                                     reference=row["reference"]),
            context_recall.ascore(user_input=row["user_input"], retrieved_contexts=contexts,
                                  reference=row["reference"]),
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
    print(results)