import re
import pymupdf4llm
import html2text
from pathlib import Path
from bs4 import BeautifulSoup
from pprint import pprint

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_chroma import Chroma
from langchain.agents import create_agent as lc_create_agent
from langchain_core.messages import AIMessageChunk
from streamlit import session_state

#----evaluations-------
from evals.ragas_eval import run_evaluation, run_agent_evaluation

#-----langfuse------
from observability import get_langfuse_handler

#----Tools-----
from tools import search_steuerbuch, create_local_search_tool

#------ui-------
import streamlit as st

#------middleware------
from langchain.agents.middleware import PIIMiddleware

#---- custom middleware
from custom_agents import TopicGuardrailMiddleware, InformationGuardrailMiddleware

# ---> Noah: Für den Agent (mehr Tokens für Reasoning + Tool-Calls)
agent_llm = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    max_tokens=300,
    temperature=0,
    #for streaming
    streaming=True
)

def load_documents() -> list[Document]:
    docs = []

    # wandelt PDFs mit pymupdf4llm in sauberes Markdown um
    for pdf_path in Path("ressources/pdf").glob("*.pdf"):
        # page_chunks=True gibt jede Seite einzeln zurück → Seitenzahl bleibt als Metadatum
        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        for i, page in enumerate(pages):
            meta = page.get("metadata", {})
            page_num = meta.get("page", meta.get("page_number", i + 1))
            docs.append(Document(
                page_content=page["text"],
                metadata={"source": pdf_path.name, "page": page_num}
            ))

    # wandelt HTML mit html2text in Markdown um
    _h2t = html2text.HTML2Text()
    _h2t.ignore_links = False
    _h2t.body_width = 0

    for html_path in Path("ressources/html").glob("*.html"):
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        # Navigation/Footer entfernen
        for tag in soup(["nav", "footer", "header", "script", "style"]):
            tag.decompose()
        main_content = soup.find("main") or soup.find("article") or soup.body
        md_text = _h2t.handle(str(main_content))
        docs.append(Document(
            page_content=md_text,
            metadata={"source": html_path.name}
        ))

    print(f"{len(docs)} Dokument-Abschnitte geladen")
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    # Pass 1: Nach Markdown-Überschriften aufteilen (## Abschnitte aus pymupdf4llm/html2text)
    # Jeder Abschnitt bekommt den Heading-Text als Metadatum (h1, h2, h3)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False  # Überschrift bleibt im Chunk-Text erhalten
    )

    # Pass 2: Jeden Abschnitt nach Zeichenlänge aufteilen falls er zu gross ist
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    for doc in docs:
        # Markdown-Header-Split: gibt Chunks mit h1/h2/h3 in metadata zurück
        header_chunks = md_splitter.split_text(doc.page_content)
        # Source + Page aus dem Original-Dokument zu jedem Chunk hinzufügen
        for chunk in header_chunks:
            chunk.metadata.update(doc.metadata)
        # Grössen-Split: lange Abschnitte weiter aufteilen
        all_chunks.extend(char_splitter.split_documents(header_chunks))

    print(f"✂️  {len(all_chunks)} Chunks erstellt")
    return all_chunks

def create_vector_store_with_embeddings():
    # create embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #create vector_store
    vector_store: Chroma = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
            collection_metadata = {"hnsw:space": "cosine"}
        )
    return vector_store, embeddings

def make_ids(chunks: list[Document]) -> list[str]:
    return [f"{c.metadata.get('source', 'unknown')}_{i}" for i, c in enumerate(chunks)]

def load_documents_into_vector_store(vector_store: Chroma, chunks: list[Document]):
    ids = make_ids(chunks)
    if vector_store.get(ids[:1])["ids"]:
        print("All chunks already in store")
    else:
        # load chunks with ids into vector store, chunks gets automatically embedded
        vector_store.add_documents(documents=chunks, ids=ids)
        print(f"{len(ids)} new chunks added to store")

def create_agent():
    # Neue Tools können hier einfach zur Liste hinzugefügt werden
    tools = [
        search_steuerbuch,  # zh.ch Steuerbuch online
    ]
    middleware = [
        # Against harmful questions related to Weapons, Bombs etc.
        TopicGuardrailMiddleware(),
        #InformationGuardrailMiddleware()
    ]

    system_prompt = (
        "Du bist ein Assistent für Steuerberatung in der Schweiz, spezialisiert auf Züricher Steuerrecht. "
        "Beantworte alle Fragen auf Deutsch.\n\n"
        "Dir werden aufbereitete lokale Dokumente als Kontext mitgegeben. "
        "Nutze NUR diese als Wissensbasis — ergänze NIEMALS aus dem Gedächtnis.\n\n"
        "TOOL-PFLICHT: Rufe search_steuerbuch auf bei:\n"
        "- Fragen zu Zürich-spezifischen Regelungen, Steuersätzen oder Fristen\n"
        "- Fragen zu konkreten ZStB-Artikeln oder dem Steuerbuch\n"
        "- Unklaren oder unvollständigen Antworten aus den lokalen Dokumenten\n"
        "Rufe das Tool IMMER auf wenn Zweifel bestehen — auch wenn lokale Dokumente vorhanden sind.\n\n"
        "Wenn keine eindeutige Antwort möglich ist, antworte: "
        "'Dazu konnte ich keine eindeutige Antwort finden. "
        "Bitte wenden Sie sich an das kantonale Steueramt Zürich.'\n\n"
        "Beziehe dich in deiner Antwort immer auf die gefundenen Quellen."
    )

    # lc_create_agent baut den kompletten Agent-Loop intern auf
    return lc_create_agent(agent_llm, tools, middleware=middleware, system_prompt=system_prompt, debug=False)

def extract_local_sources(rag_context: str) -> list[str]:
    # Parst den RAG-Output nach Quelle: datei.pdf, Seite X, Abschnitt Einträgen
    # Seite und Abschnitt sind optional (HTML-Dateien haben keine Seitenzahl)
    matches = re.findall(
        r'\[Quelle: ([^,\]]+)(?:, Seite ([^,\]]+))?(?:, ([^\]]+))?\]',
        rag_context
    )
    sources = []
    for file, page, section in matches:
        entry = file.strip()
        if page:
            entry += f" (Seite {page.strip()}"
            if section:
                entry += f", {section.strip()}"
            entry += ")"
        elif section:
            entry += f" ({section.strip()})"
        sources.append(entry)
    return list(dict.fromkeys(sources))  # Duplikate entfernen, Reihenfolge beibehalten

def extract_steuerbuch_urls(messages: list) -> list[str]:
    # Parst ToolMessages des Agents nach zh.ch-URLs aus dem search_steuerbuch Tool
    urls = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "search_steuerbuch":
            matches = re.findall(r'Quelle: (https://[^\n]+)', msg.content)
            urls.extend(m.strip() for m in matches)
    return list(dict.fromkeys(urls))

#neu hinzugefügt werden -> spart ressourcen
@st.cache_resource(hash_funcs={Chroma: lambda _: None})
def get_cached_agent(vector_store):
    return create_agent(), create_local_search_tool(vector_store)

#For run_agent_evaluation purposes
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

#For UI purposese
def get_agent_response_stream(query, vector_store, chat_history):
    agent, local_search = get_cached_agent(vector_store)
    rag_context = local_search.invoke({"query": query})
    messages = chat_history + [HumanMessage(
        content=f"Lokale Dokumente:\n{rag_context}\n\nFrage: {query}"
    )]

    def stream_generator():
        for chunk in agent.stream(
                {"messages": messages},
                config={"callbacks": [get_langfuse_handler()]},
                stream_mode="messages"  # ← Token-by-Token Modus
        ):
            if isinstance(chunk, tuple):
                #for logging
                message, metadata = chunk
                pprint(f"Message: {message}\n Metadata: {metadata}")
                if hasattr(message, "content") and message.content and metadata.get("langgraph_node") == 'model':
                    if isinstance(message, AIMessageChunk):
                        yield message.content

    return stream_generator(), rag_context

def start_agent(vector_store: Chroma):
    st.title("Steuer-Assistenz")
    st.info("Die Antworten stellen keine rechtsverbindliche Auskunft dar und können Fehler enthalten und kantonal variieren. Es wird keinerlei Haftung übernommen.", icon="🚨", width="stretch")

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
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Der EINZIGE Chat Input
    if prompt := st.chat_input("Wie kann ich helfen?", key="main_chat"):
        # User-Eingabe anzeigen & speichern
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)


        # 4. Logik ausführen
        with st.chat_message("assistant", avatar='⚖️'):
            result = None
            rag_context = ""

            try:
                stream, rag_context = get_agent_response_stream(
                    prompt,
                    vector_store,
                    st.session_state.chat_history
                )
                answer = st.write_stream(stream)
            except ValueError as e:
                answer = str(e)
                rag_context = ""
                result = None
                st.markdown(answer)

            # Lokale Quellen aus RAG anzeigen
            local_sources = extract_local_sources(rag_context)
            steuerbuch_urls = extract_steuerbuch_urls(result["messages"] if result else [])
            with st.expander("Quellen"):
                if local_sources:
                    st.caption(f"Quellen: \n\n {', '.join(local_sources)}")
                if steuerbuch_urls:
                    st.caption("Steuerbuch-Quellen: \n\n " + " · ".join(steuerbuch_urls))


        # 5. Historie aktualisieren
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=answer),
        ])

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

def main():
    # create Vector Store with Embeddings (_ ist die Python-Konvention für "wird nicht gebraucht)
    vector_store, _ = create_vector_store_with_embeddings()

    # Nur laden und splitten, wenn die DB noch leer ist
    if vector_store.get(limit=1)["ids"]:
        print("✅ Vector Store bereits befüllt — Laden und Splitting übersprungen")
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
