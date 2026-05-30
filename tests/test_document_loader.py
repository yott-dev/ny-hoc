
from unittest.mock import patch, MagicMock
from src.document_loader import load_documents, split_documents
from langchain_core.documents import Document

#splittet da page_content > 30 chars
def test_split_documents_keeps_long_chunks():
    docs = [Document(
        page_content="Das ist ein Satz über Steuern" * 100,  # 100 Zeichen, eindeutig über 30
        metadata={"folder": "Tabellen"}
    )]
    result = split_documents(docs)
    assert len(result) >= 1  # mindestens ein Chunk bleibt

#splittet nicht da page_content < 30 chars
def test_split_documents_filters_short_chunks():
    docs = [Document(
        page_content="Das ist",  # unter 30 Zeichen
        metadata={"folder": "Tabellen"}
    )]
    result = split_documents(docs)
    assert result == []  # wird herausgefiltert

#wenn folder = tabelle, dann wird nur header gesplittet nicht tabelle selbst
def test_split_documents_tabellen_not_split():
    langer_text = "x" * 1000  # über chunk_size=750
    docs = [Document(
        page_content=langer_text,
        metadata={"folder": "Tabellen"}
    )]
    result = split_documents(docs)
    assert len(result) == 1  # bleibt ein Chunk

#wenn folder not tabelle dann wird normal gesplittet
def test_split_documents_normal_gets_split():
    langer_text = ("Das ist ein Satz über Steuern. " * 50)
    docs = [Document(
        page_content=langer_text,
        metadata={"folder": "pdf"}
    )]
    result = split_documents(docs)
    assert len(result) > 1

def test_split_documents_metadata_is_preserved():
    docs = [Document(
        page_content="Das ist ein Satz über Steuern. " * 10,
        metadata={"source": "steuern.pdf", "page": 3}
    )]
    result = split_documents(docs)
    assert result[0].metadata["source"] == "steuern.pdf"
    assert result[0].metadata["page"] == 3

#splittet in zwei chunks, da zwei Titel
def test_split_documents_splits_on_markdown_headers():
    text = (
        "# Titel\nDieser Abschnitt handelt von Einkommenssteuern in Zürich.\n\n"
        "## Untertitel\nDieser Abschnitt handelt von Abzügen und Freibeträgen."
    )
    docs = [Document(page_content=text, metadata={"folder": "pdf"})]
    result = split_documents(docs)
    assert len(result) >= 2

#============================================================

def test_load_documents_pdf():
    fake_path = MagicMock()
    fake_path.name = "steuern.pdf"
    fake_path.parent.name = "pdf"
    fake_pages = [{"text": "Das ist Seiteninhalt.", "metadata": {"page": 1}}]

    with patch("src.document_loader.Path.rglob", return_value=[fake_path]), \
         patch("src.document_loader.Path.glob", return_value=[]), \
         patch("src.document_loader.pymupdf4llm.to_markdown", return_value=fake_pages):
        result = load_documents()

    assert len(result) == 1
    assert result[0].metadata["source"] == "steuern.pdf"

def test_load_documents_html():
    fake_html_path = MagicMock()
    fake_html_path.name = "steuern.html"
    fake_html_path.read_text.return_value = "<html><body><p>Steuerinhalt</p></body></html>"

    with patch("src.document_loader.Path.rglob", return_value=[]), \
         patch("src.document_loader.Path.glob", side_effect=[[fake_html_path], []]), \
         patch("src.document_loader.pymupdf4llm.to_markdown", return_value=[]):
        result = load_documents()

    assert len(result) == 1
    assert result[0].metadata["source"] == "steuern.html"


def test_load_documents_empty():
    with patch("src.document_loader.Path.rglob", return_value=[]), \
            patch("src.document_loader.Path.glob", side_effect=[[], []]):
        result = load_documents()

    assert result == []