from src.utils_parsing import extract_local_sources, make_ids, extract_steuerbuch_urls
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
import hashlib

# Fall 1: Mit Seitenzahl
def test_extract_local_sources_with_page():
    result = extract_local_sources("[Quelle: steuern.pdf, Seite 5]")
    assert result == ["steuern.pdf (Seite 5)"]

# Fall 2: Ohne Seitenzahl
def test_extract_local_sources_without_page():
    result = extract_local_sources("[Quelle: steuern.pdf]")
    assert result == ["steuern.pdf"]

# Fall 3: Leerer String
def test_extract_local_sources_empty():
    result = extract_local_sources("")
    assert result == []


#==================================================


def test_make_ids_with_metadata():
    result = make_ids([Document(page_content="This is the page content", metadata={"source": "page9"})])
    assert result == ["page9_0"]

def test_make_ids_without_metadata():
    chunk = Document(page_content="This is the page content")
    expected_hash = hashlib.md5("This is the page content".encode()).hexdigest()[:8]
    result = make_ids([chunk])
    assert result == [f"unknown_{expected_hash}"]

def test_make_ids_multiple_chunks():
    chunks = [
        Document(page_content="Text 1", metadata={"source": "a.pdf"}),
        Document(page_content="Text 2", metadata={"source": "b.pdf"}),
    ]
    hash1 = hashlib.md5("Text 1".encode()).hexdigest()[:8]
    hash2 = hashlib.md5("Text 2".encode()).hexdigest()[:8]
    result = make_ids(chunks)
    assert result == [f"a.pdf_{hash1}", f"b.pdf_{hash2}"]

def test_make_ids_without_anything():
    result = make_ids([])
    assert result == []

def test_make_ids_with_metadata():
    chunk = Document(page_content="This is the page content", metadata={"source": "page9"})
    expected_hash = hashlib.md5("This is the page content".encode()).hexdigest()[:8]
    result = make_ids([chunk])
    assert result == [f"page9_{expected_hash}"]

#=========================================================

def test_extract_steuerbuch_urls_with_content():
    result = extract_steuerbuch_urls([ToolMessage(content="Quelle: https://zh_steuern", tool_call_id="1", name="search_steuerbuch")])
    assert result == ["https://zh_steuern"]

def test_extract_steuerbuch_urls_without_content():
    result = extract_steuerbuch_urls([ToolMessage(content="", tool_call_id="1", name="search_steuerbuch")])
    assert result == []

def test_extract_steuerbuch_urls_with_wrong_name():
    result = extract_steuerbuch_urls([ToolMessage(content="Quelle: https://zh_steuern", tool_call_id="1", name="wrong_tool")])
    assert result == []

def test_extract_steuerbuch_urls_without_anything():
    result = extract_steuerbuch_urls([])
    assert result == []

def test_extract_steuerbuch_urls_duplicates():
    msg = ToolMessage(
        content="Quelle: https://zh.ch/steuer\nQuelle: https://zh.ch/steuer",
        name="search_steuerbuch",
        tool_call_id="1"
    )
    result = extract_steuerbuch_urls([msg])
    assert result == ["https://zh.ch/steuer"]  # ← nur einmal

def test_get_langfuse_handler_returns_handler():
    from src.utils_parsing import get_langfuse_handler
    result = get_langfuse_handler()
    assert result is not None

