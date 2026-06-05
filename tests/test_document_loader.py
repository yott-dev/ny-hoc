from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.document_loader import load_documents, split_documents

# ─── DOCUMENT SPLITTING & CHUNKING TESTS ──────────────────────────────────────

def test_split_documents_keeps_long_chunks():
    """Ensures text contents exceeding the minimum threshold (30 chars) are preserved."""
    docs = [Document(
        page_content="Das ist ein Satz über Steuern" * 10,  # Explicitly longer than 30 chars
        metadata={"folder": "Tabellen"}
    )]
    result = split_documents(docs)
    assert len(result) >= 1


def test_split_documents_filters_short_chunks():
    """Verifies that noise or short fragments under 30 characters are filtered out."""
    docs = [Document(
        page_content="Das ist",  # Under 30 characters
        metadata={"folder": "Tabellen"}
    )]
    result = split_documents(docs)
    assert result == []


def test_split_documents_tabellen_not_split():
    """Documents from the 'Tabellen' folder must remain intact as a single chunk

    to preserve structural data row-alignments.
    """
    long_text = "x" * 1000  # Exceeds the standard chunk_size limit of 750
    docs = [Document(
        page_content=long_text,
        metadata={"folder": "Tabellen"}
    )]
    result = split_documents(docs)
    assert len(result) == 1


def test_split_documents_normal_gets_split():
    """Standard documents (e.g., PDFs) exceeding chunk limits must be split into multiple items."""
    long_text = ("Das ist ein Satz über Steuern. " * 50)
    docs = [Document(
        page_content=long_text,
        metadata={"folder": "pdf"}
    )]
    result = split_documents(docs)
    assert len(result) > 1


def test_split_documents_metadata_is_preserved():
    """Ensures critical operational metadata passes down to final splits without loss."""
    docs = [Document(
        page_content="Das ist ein Satz über Steuern. " * 10,
        metadata={"source": "steuern.pdf", "page": 3}
    )]
    result = split_documents(docs)
    assert result[0].metadata["source"] == "steuern.pdf"
    assert result[0].metadata["page"] == 3


def test_split_documents_splits_on_markdown_headers():
    """Validates that the splitter segments content based on Markdown structural headings."""
    text = (
        "# Titel\nDieser Abschnitt handelt von Einkommenssteuern in Zürich.\n\n"
        "## Untertitel\nDieser Abschnitt handelt von Abzügen und Freibeträgen."
    )
    docs = [Document(page_content=text, metadata={"folder": "pdf"})]
    result = split_documents(docs)
    assert len(result) >= 2


# ─── CORE DOCUMENT LOADING & FILESYSTEM TESTS ─────────────────────────────────

def test_load_documents_pdf():
    """Mocks the filesystem and PyMuPDF4LLM to simulate reading a localized PDF document."""
    fake_path = MagicMock()
    fake_path.name = "steuern.pdf"
    fake_path.parent.name = "pdf"
    fake_pages = [{"text": "Das ist Seiteninhalt.", "metadata": {"page": 1}}]

    # We patch pathlib.Path directly at the source instead of via the local module path
    with patch("pathlib.Path.rglob", return_value=[fake_path]), \
         patch("pathlib.Path.glob", return_value=[]), \
         patch("src.document_loader.pymupdf4llm.to_markdown", return_value=fake_pages):
        result = load_documents()

    assert len(result) == 1
    assert result[0].metadata["source"] == "steuern.pdf"


def test_load_documents_html():
    """Mocks filesystem hooks to simulate parsing an HTML source document file."""
    fake_html_path = MagicMock()
    fake_html_path.name = "steuern.html"
    fake_html_path.read_text.return_value = "<html><body><p>Steuerinhalt</p></body></html>"

    with patch("pathlib.Path.rglob", return_value=[]), \
         patch("pathlib.Path.glob", side_effect=[[fake_html_path], []]), \
         patch("src.document_loader.pymupdf4llm.to_markdown", return_value=[]):
        result = load_documents()

    assert len(result) == 1
    assert result[0].metadata["source"] == "steuern.html"


def test_load_documents_empty():
    """Ensures robust handling and empty array return when zero matching files are present."""
    with patch("pathlib.Path.rglob", return_value=[]), \
         patch("pathlib.Path.glob", side_effect=[[], []]):
        result = load_documents()

    assert result == []