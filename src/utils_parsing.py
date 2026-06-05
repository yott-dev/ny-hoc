import re
import hashlib
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage, BaseMessage
from langfuse.langchain import CallbackHandler

load_dotenv()


def get_langfuse_handler() -> CallbackHandler:
    """
    Initializes and returns the Langfuse tracing callback handler.
    """
    return CallbackHandler()


def extract_local_sources(rag_context: str) -> list[str]:
    """
    Parses structural metadata tags from raw RAG context blocks.
    Expected format: '[Quelle: filename.pdf, Seite X, Section Y]'
    Note: Page and section fields are treated as optional patterns.
    """
    # Regex breakdown: Group 1 = File, Group 2 = Optional Page, Group 3 = Optional Section
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

    # Convert dict keys to list to strip out duplicates while preserving original order
    return list(dict.fromkeys(sources))


def extract_steuerbuch_urls(messages: list[BaseMessage]) -> list[str]:
    """
    Extracts verification URLs from official ToolMessages emitted by the search_steuerbuch tool.
    """
    urls = []
    for msg in messages:
        # We only target ToolMessages produced by our custom live scaper tool
        if isinstance(msg, ToolMessage) and msg.name == "search_steuerbuch":
            matches = re.findall(r'Quelle: (https://[^\n]+)', msg.content)
            urls.extend(m.strip() for m in matches)

    return list(dict.fromkeys(urls))


def make_ids(chunks: list[Document]) -> list[str]:
    """
    Generates deterministic MD5 hashes to serve as stable vector store chunk IDs.
    Combines source pathing, page metadata, and core page content strings
    to protect against key collisions.
    """
    return [
        hashlib.md5(
            f"{c.metadata.get('source', 'unknown')}|{c.metadata.get('page', i)}|{c.page_content}".encode()
        ).hexdigest()
        for i, c in enumerate(chunks)
    ]