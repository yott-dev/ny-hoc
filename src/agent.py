from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain.agents import create_agent as lc_create_agent
from src.config import agent_llm
from langchain_core.messages import AIMessageChunk

#-----langfuse------
from src.utils_parsing import get_langfuse_handler

#----Tools-----
from tools import search_steuerbuch, create_local_search_tool

#------ui-------
import streamlit as st

#---- custom middleware
from src.custom_agents import TopicGuardrailMiddleware

def create_agent():
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

    # lc_create_agent baut den kompletten Agent-Loop intern auf
    return lc_create_agent(agent_llm, tools, middleware=middleware, system_prompt=system_prompt, debug=False)

@st.cache_resource(hash_funcs={Chroma: lambda _: None})
def get_cached_agent(vector_store):
    return create_agent(), create_local_search_tool(vector_store)

def get_agent_response(query, vector_store, chat_history):
    agent, local_search = get_cached_agent(vector_store)
    rag_context = local_search.invoke({"query": query})
    messages = chat_history + [HumanMessage(
        content=f"Lokale Dokumente:\n{rag_context}\n\nFrage: {query}"
    )]

    result = agent.invoke(
        {"messages": messages},
        config={"callbacks": [get_langfuse_handler()]}
    )
    return result, rag_context


def get_agent_response_stream(query, vector_store, chat_history):
    agent, local_search = get_cached_agent(vector_store)
    rag_context = local_search.invoke({"query": query})
    messages = chat_history + [HumanMessage(
        content=f"Lokale Dokumente:\n{rag_context}\n\nFrage: {query}"
    )]

    collected_messages = []  # sammelt alle Messages während dem Stream für URL-Extraktion

    def stream_generator():
        for chunk in agent.stream(
                {"messages": messages},
                #config={"callbacks": [get_langfuse_handler()]},
                stream_mode="messages"  # ← Token-by-Token Modus
        ):
            if isinstance(chunk, tuple):
                #for logging
                message, metadata = chunk
                collected_messages.append(message)
                #pprint(f"Message: {message}\n Metadata: {metadata}")
                if hasattr(message, "content") and message.content and metadata.get("langgraph_node") == 'model':
                    if isinstance(message, AIMessageChunk):
                        yield message.content

    return stream_generator(), rag_context, collected_messages