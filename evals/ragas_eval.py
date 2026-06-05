import os
import re
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Any
from openai import AsyncOpenAI

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ragas import experiment, Dataset
from ragas.testset import TestsetGenerator
from ragas.metrics.collections import AnswerAccuracy, Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

# Global metric inventory schema for historical plotting tracking routines
METRICS = ["answer_accuracy", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def ask_chatbot(query: str, vector_store: Chroma, llm: BaseChatModel) -> tuple[str, list[str]]:
    """
    Executes a basic baseline RAG chain execution and returns the text response and retrieved context blocks.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    retrieve_data = retriever.invoke(query)
    contexts = [doc.page_content for doc in retrieve_data]

    template = (
        "Du bist ein Assistent für Steuerberatung in der Schweiz, spezialisiert auf Züricher Steuerrecht.\n "
        "Bei widersprüchlichen Informationen: Priorisiere Dokumente mit neuerem Jahr"
        "Beantworte alle Fragen auf Deutsch.\n\n"
        "Dir werden aufbereitete lokale Dokumente als Kontext mitgegeben. "
        "Nutze NUR diese als Wissensbasis — ergänze NIEMALS aus dem Gedächtnis.\n\n"
        "Rufe das Tool IMMER auf wenn Zweifel bestehen — auch wenn lokale Dokumente vorhanden sind.\n\n"
        "Wenn keine eindeutige Antwort möglich ist, antworte: "
        "'Dazu konnte ich keine eindeutige Antwort finden. "
        "Bitte wenden Sie sich an das kantonale Steueramt Zürich.'\n\n"
        "Beziehe dich in deiner Antwort immer auf die gefundenen Quellen."
        "Kontext: {context}"
        "Frage: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    answer = rag_chain.invoke(query)
    return answer, contexts


def generate_testset(docs: list[Document], ragas_llm: Any, ragas_embeddings: Any,
                     testset_size: int = 10) -> pd.DataFrame:
    """
    Generates a synthetic testing evaluation set from provided LangChain text documents.
    """
    csv_path = "evals/datasets/data.csv"

    if os.path.exists(csv_path):
        print("✅ Testset already exists locally. Loading file...")
        return pd.read_csv(csv_path)

    print("🔄 Generating a new synthetic Ragas testset tracking frame...")

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
    print("✅ Synthetic testset stored successfully.")

    return df


def load_dataset() -> Dataset:
    """
    Loads the persistent evaluation dataset frame from local storage boundaries.
    """
    dataset = Dataset.load(
        name="data_v5",
        backend="local/csv",
        root_dir="./evals"
    )
    return dataset


def next_experiment_name(base: str, folder: str = "evals/experiments") -> str:
    """
    Returns the next incremental iteration version tracking name (e.g., steuerbot_v3).
    """
    existing = [f for f in os.listdir(folder) if re.match(rf"{re.escape(base)}_v\d+\.csv", f)]
    return f"{base}_v{len(existing) + 1}"


def plot_experiments(base: str = "steuerbot_agent", folder: str = "evals/experiments", save: bool = False) -> None:
    """
    Loads all saved historical experiment csv metrics and plots development trends via a line graph.
    """
    files = sorted(f for f in os.listdir(folder) if re.match(rf"{re.escape(base)}_v\d+\.csv", f))
    if not files:
        print("❌ No historical experiment datasets found matching specifications.")
        return

    records = []
    for fname in files:
        df = pd.read_csv(os.path.join(folder, fname))
        row = {"version": fname.removesuffix(".csv")}
        for m in METRICS:
            if m in df.columns:
                row[m] = round(df[m].mean(), 3)
        records.append(row)

    summary = pd.DataFrame(records).set_index("version")
    print("\n📈 Aggregated Experiment Evaluation History Summary:")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in METRICS:
        if metric in summary.columns:
            ax.plot(summary.index, summary[metric], marker="o", label=metric)

    ax.set_title(f"Evaluation Metrics Comparison Run Trend: {base}")
    ax.set_xlabel("Experiment Incremental Runs")
    ax.set_ylabel("Score Limits (0.00 – 1.00)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if save:
        save_path = os.path.join(folder, f"{base}_verlauf.png")
        plt.savefig(save_path, dpi=150)
        print(f"📊 Tracking comparison visualization exported to: {save_path}")

    plt.show()


def run_evaluation(vector_store: Chroma) -> None:
    """
    Runs an evaluation batch test on the vanilla baseline RAG application chain layout.
    """
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o", client=client)
    langchain_llm = init_chat_model("gpt-4o", model_provider="openai")
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-large",
        client=client,
        interface="modern"
    )

    answer_accuracy = AnswerAccuracy(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    dataset = load_dataset()

    @experiment()
    async def run_experiment(row: dict[str, Any]) -> dict[str, Any]:
        answer, contexts = ask_chatbot(row["user_input"], vector_store, langchain_llm)

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
            "user_input": row["user_input"],
            "reference": row["reference"],
            "response": answer,
            "answer_accuracy": accuracy.value,
            "faithfulness": faith.value,
            "answer_relevancy": relevancy.value,
            "context_precision": precision.value,
            "context_recall": recall.value,
        }

    name = next_experiment_name("steuerbot")
    print(f"💾 Saving evaluation database file under moniker: {name}")
    results = asyncio.run(run_experiment.arun(dataset, name=name))
    print(results.to_pandas().to_string())
    plot_experiments(base="steuerbot")


def run_agent_evaluation(vector_store: Chroma, agent_fn: Callable[
    [str, Chroma, list[BaseMessage]], tuple[dict[str, Any], str, list[Document]]]) -> None:
    """
    Runs an evaluation batch test capturing the custom agent graph orchestration workflow metrics.
    """
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o", client=client)
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-large",
        client=client,
        interface="modern"
    )

    answer_accuracy = AnswerAccuracy(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    dataset = load_dataset()

    @experiment()
    async def run_experiment(row: dict[str, Any]) -> dict[str, Any]:
        result, rag_context, rag_docs = agent_fn(row["user_input"], vector_store, [])
        answer = result["messages"][-1].content

        # Feed Ragas raw page chunks alongside executed ToolMessage contents to evaluate compound grounding
        doc_contexts = [doc.page_content for doc in rag_docs]
        tool_texts = [msg.content for msg in result["messages"] if isinstance(msg, ToolMessage)]
        contexts = doc_contexts + tool_texts

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
            "user_input": row["user_input"],
            "reference": row["reference"],
            "response": answer,
            "answer_accuracy": accuracy.value,
            "faithfulness": faith.value,
            "answer_relevancy": relevancy.value,
            "context_precision": precision.value,
            "context_recall": recall.value,
        }

    name = next_experiment_name("steuerbot_agent")
    print(f"💾 Saving evaluation agent database file under moniker: {name}")
    results = asyncio.run(run_experiment.arun(dataset, name=name))
    print(results.to_pandas().to_string())
    plot_experiments(base="steuerbot_agent")


if __name__ == "__main__":
    plot_experiments(base="steuerbot_agent", save=True)