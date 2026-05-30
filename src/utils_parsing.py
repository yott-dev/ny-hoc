import re
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langfuse.langchain import CallbackHandler
import hashlib


def get_langfuse_handler() -> CallbackHandler:
    return CallbackHandler()

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
        else:
            urls.extend("")
    return list(dict.fromkeys(urls))

#def make_ids(chunks: list[Document]) -> list[str]:
    return [f"{c.metadata.get('source', 'unknown')}_{i}" for i, c in enumerate(chunks)]

def make_ids(chunks: list[Document]) -> list[str]:
    return [
        f"{c.metadata.get('source', 'unknown')}_{hashlib.md5(c.page_content.encode()).hexdigest()[:8]}"
        for c in chunks
    ]