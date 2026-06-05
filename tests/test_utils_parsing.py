import hashlib
import re
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from src.utils_parsing import extract_local_sources, make_ids, extract_steuerbuch_urls


# ─── LOCAL SOURCE EXTRACTION TESTS ─────────────────────────────────────────────

def test_extract_local_sources_with_page():
    result = extract_local_sources("[Quelle: steuern.pdf, Seite 5]")
    assert result == ["steuern.pdf (Seite 5)"]


def test_extract_local_sources_without_page():
    result = extract_local_sources("[Quelle: steuern.pdf]")
    assert result == ["steuern.pdf"]


def test_extract_local_sources_empty():
    result = extract_local_sources("")
    assert result == []


# ─── DETERMINISTIC ID GENERATION TESTS ─────────────────────────────────────────

def test_make_ids_with_metadata():
    """Validates that the generated ID is a valid deterministic hex hash string."""
    chunk = Document(
        page_content="This is the page content that will be chunked and indexed.",
        metadata={"source": "page9", "page": 9}
    )
    result = make_ids([chunk])

    assert len(result) == 1
    # Asserts that the ID is a hex string (lowercase letters a-f and numbers 0-9)
    assert re.match(r"^[a-f0-9]+$", result[0]) is not None


def test_make_ids_without_metadata():
    """Validates hex hash generation fallback when metadata fields are omitted."""
    chunk = Document(page_content="This is the page content without any metadata fields.")
    result = make_ids([chunk])

    assert len(result) == 1
    assert re.match(r"^[a-f0-9]+$", result[0]) is not None


def test_make_ids_multiple_chunks():
    """Validates that multiple unique chunks result in distinct hex hash identifiers."""
    chunks = [
        Document(page_content="Text 1 content snippet", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Text 2 content snippet", metadata={"source": "b.pdf", "page": 2}),
    ]
    result = make_ids(chunks)

    assert len(result) == 2
    assert result[0] != result[1]  # Ensure uniqueness
    assert re.match(r"^[a-f0-9]+$", result[0]) is not None
    assert re.match(r"^[a-f0-9]+$", result[1]) is not None


def test_make_ids_without_anything():
    result = make_ids([])
    assert result == []


# ─── TAX BOOK URL EXTRACTION TESTS ─────────────────────────────────────────────

def test_extract_steuerbuch_urls_with_content():
    result = extract_steuerbuch_urls([
        ToolMessage(content="Quelle: https://zh_steuern", tool_call_id="1", name="search_steuerbuch")
    ])
    assert result == ["https://zh_steuern"]


def test_extract_steuerbuch_urls_without_content():
    result = extract_steuerbuch_urls([
        ToolMessage(content="", tool_call_id="1", name="search_steuerbuch")
    ])
    assert result == []


def test_extract_steuerbuch_urls_with_wrong_name():
    result = extract_steuerbuch_urls([
        ToolMessage(content="Quelle: https://zh_steuern", tool_call_id="1", name="wrong_tool")
    ])
    assert result == []


def test_extract_steuerbuch_urls_without_anything():
    result = extract_steuerbuch_urls([])
    assert result == []


def test_extract_steuerbuch_urls_duplicates():
    """Ensures duplicate URLs parsed within tool responses are deduplicated into a unique list."""
    msg = ToolMessage(
        content="Quelle: https://zh.ch/steuer\nQuelle: https://zh.ch/steuer",
        name="search_steuerbuch",
        tool_call_id="1"
    )
    result = extract_steuerbuch_urls([msg])
    assert result == ["https://zh.ch/steuer"]


def test_get_langfuse_handler_returns_handler():
    from src.utils_parsing import get_langfuse_handler
    result = get_langfuse_handler()
    assert result is not None