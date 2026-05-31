from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from src.utils_parsing import extract_local_sources, extract_steuerbuch_urls
from src.agent import get_agent_response_stream
from src.config import MAX_HISTORY_PAIRS, agent_llm
from langchain.agents import create_agent

#------ui-------
import streamlit as st

def start_agent(vector_store: Chroma):
    st.title("Steuer-Assistenz")
    st.info("Die Antworten stellen keine rechtsverbindliche Auskunft dar und können Fehler enthalten und kantonal variieren. Es wird keinerlei Haftung übernommen.", width="stretch")

    #load logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # 1. Session State initialisieren (global)
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Liste aus Dictionaries für UI
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Liste aus Message-Objekten für Agent

    # 2. Chat Verlauf laden, wenn noch nicht vorhanden
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    # 3. Der EINZIGE Chat Input
    if prompt := st.chat_input("Wie kann ich helfen?", key="main_chat"):
        # User-Eingabe anzeigen & speichern
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": '👤'})
        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)


        # 4. Logik ausführen
        with st.chat_message("assistant", avatar='⚖️'):
            result = None
            rag_context = ""

            try:
                stream, rag_context, collected_messages = get_agent_response_stream(
                    prompt,
                    vector_store,
                    st.session_state.chat_history
                )
                answer = st.write_stream(stream)

                # Lokale Quellen aus RAG anzeigen
                local_sources = extract_local_sources(rag_context)
                steuerbuch_urls = extract_steuerbuch_urls(collected_messages)
                with st.expander("Quellen"):
                    if local_sources:
                        st.caption("**Lokale Quellen:**\n\n" + "\n\n".join(f"• {s}" for s in local_sources))
                    if steuerbuch_urls:
                        st.caption("**Steuerbuch-Quellen:**\n\n" + "\n\n".join(f"• {u}" for u in steuerbuch_urls))

            except ValueError as e:
                answer = str(e)
                rag_context = ""
                collected_messages = []
                st.markdown(answer)

        # 5. Historie aktualisieren
        st.session_state.messages.append({"role": "assistant", "content": answer, "avatar": "⚖️"})
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=answer),
        ])
        # Rolling Window: nur letzte MAX_HISTORY_PAIRS Paare für den Agent behalten
        # (UI-messages bleibt vollständig, damit der User zurückscrollen kann)
        st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_PAIRS * 2):]


def start_authentication(vector_store):
    #initialisiere session_state.authenticated
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Credentials aus secrets.toml laden
    if not st.session_state.authenticated:
        st.title("Login")
        username = st.text_input("Benutzername", type="default")
        password = st.text_input("Passwort", type="password")

        if st.button("Login"):
            if (username == st.secrets["credentials"]["username"] and
                    password == st.secrets["credentials"]["password"]):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Falscher Benutzername oder Passwort")
        return
    # Wenn Login gut, lade agent
    start_agent(vector_store)

