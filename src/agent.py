from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain.agents import create_agent as lc_create_agent
from src.config import agent_llm
from langchain_core.messages import AIMessageChunk
from src.utils_parsing import get_langfuse_handler
from src.tools import search_steuerbuch
from src.retriever import retrieve_local_context
import streamlit as st
from src.custom_agents import TopicGuardrailMiddleware

def create_agent():
    """Compiles the LangGraph tax agent with active tools and guardrail middleware."""
    # Neue Tools können hier einfach zur Liste hinzugefügt werden
    tools = [
        search_steuerbuch,  # zh.ch Steuerbuch online
    ]
    middleware = [
        # Against harmful questions related to Weapons, Bombs etc.
        TopicGuardrailMiddleware(),
    ]

    system_prompt = (
    "Du bist ein Steuerberater-Assistent für Züricher Steuerrecht. Antworte auf Deutsch.\n\n"

    "TOOL-AUFRUF (search_steuerbuch):\n"
    "- Aufrufen bei: kantonalen Zürich-spezifischen Regelungen, Steuertarifen, Abzugslimiten, gesetzlichen Fristen.\n"
    "- Nicht nötig bei: allgemeinen Definitionen die in den lokalen Dokumenten vollständig abgedeckt sind.\n\n"

    "ANTWORT-REGELN:\n"
    "- Nutze NUR die mitgegebenen lokalen Dokumente und search_steuerbuch, ergänze NIEMALS aus dem Gedächtnis.\n"
    "- Bei widersprüchlichen Quellen: bevorzuge das neuere Dokument.\n"
    "- Gib immer die Quellen an (Datei, Seite oder URL).\n"
    "- Wenn keine eindeutige Antwort möglich: antworte mit "
    "'Dazu konnte ich keine eindeutige Antwort finden. "
    "Bitte wenden Sie sich an das kantonale Steueramt Zürich.'\n"
    )

    return lc_create_agent(agent_llm, tools, middleware=middleware, system_prompt=system_prompt, debug=False)

@st.cache_resource(hash_funcs={Chroma: lambda _: None})
def get_cached_agent(vector_store: Chroma):
    """Caches the compiled agent graph in Streamlit using the vector store as a cache key dependency."""
    return create_agent()

def get_agent_response(query: str, vector_store: Chroma, chat_history:list) -> tuple:
    """Executes a synchronous, blocking invocation of the agent with injected RAG context."""
    agent = get_cached_agent(vector_store)
    rag_context, rag_docs = retrieve_local_context(query, vector_store)
    messages = chat_history + [HumanMessage(
        content=f"Lokale Dokumente:\n{rag_context}\n\nFrage: {query}"
    )]

    result = agent.invoke(
        {"messages": messages},
        config={"callbacks": [get_langfuse_handler()]}
    )
    return result, rag_context, rag_docs


def get_agent_response_stream(query: str, vector_store: Chroma, chat_history: list) -> list:
    """Yields real-time text tokens from the agent's model node during execution."""
    agent = get_cached_agent(vector_store)
    rag_context, _ = retrieve_local_context(query, vector_store)
    messages = chat_history + [HumanMessage(
        content=f"Lokale Dokumente:\n{rag_context}\n\nFrage: {query}"
    )]

    collected_messages = []

    def stream_generator():
        for chunk in agent.stream(
                {"messages": messages},
                config={"callbacks": [get_langfuse_handler()]},
                stream_mode="messages"
        ):
            if isinstance(chunk, tuple):
                message, metadata = chunk
                collected_messages.append(message)
                if hasattr(message, "content") and message.content and metadata.get("langgraph_node") == 'model':
                    if isinstance(message, AIMessageChunk):
                        yield message.content

    return stream_generator(), rag_context, collected_messages