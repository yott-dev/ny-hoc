import re
import pymupdf4llm
from dotenv import load_dotenv
load_dotenv()
import html2text
from pathlib import Path
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

def load_documents() -> list[Document]:
    docs = []

    # wandelt PDFs mit pymupdf4llm in sauberes Markdown um
    for pdf_path in Path("ressources/pdf").rglob("*.pdf"):
        # page_chunks=True gibt jede Seite einzeln zurück → Seitenzahl bleibt als Metadatum
        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        for i, page in enumerate(pages):
            meta = page.get("metadata", {})
            page_num = meta.get("page", meta.get("page_number", i + 1))
            docs.append(Document(
                page_content=page["text"],
                metadata={"source": pdf_path.name, "page": page_num, "folder": pdf_path.parent.name}
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

    # Markdown-Dateien direkt einlesen (keine Konvertierung nötig)
    for md_path in Path("ressources/md").glob("*.md"):
        docs.append(Document(
            page_content=md_path.read_text(encoding="utf-8"),
            metadata={"source": md_path.name}
        ))

    print(f"{len(docs)} Dokument-Abschnitte geladen")
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    # Pass 1: Nach Markdown-Überschriften aufteilen (## Abschnitte aus pymupdf4llm/html2text)
    # Jeder Heading-Text (h1, h2, h3) wird pro Abschnitt in die Metadaten geschrieben
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False  # Überschrift bleibt im Chunk-Text erhalten
    )

    # Pass 2: Jeden Abschnitt nach Zeichenlänge aufteilen falls er zu gross ist
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=250,
        #verhindert, dass mitten in einem wort getrennt wird
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    for doc in docs:
        # Markdown-Header-Split: gibt Chunks mit h1/h2/h3 in metadata zurück
        header_chunks = md_splitter.split_text(doc.page_content)
        # Source + Page aus dem Original-Dokument zu jedem Chunk hinzufügen
        for chunk in header_chunks:
            chunk.metadata.update(doc.metadata)

        # Tabellen-PDFs (Unterordner "Tabellen/") als ganze Seite behalten —
        # der Char-Splitter würde Tabellenzeilen vom Header trennen und den Kontext zerstören
        if doc.metadata.get("folder") == "Tabellen":
            all_chunks.extend(header_chunks)
        else:
            # Grössen-Split: lange Abschnitte weiter aufteilen
            all_chunks.extend(char_splitter.split_documents(header_chunks))

    # Leere und zu kurze Chunks entfernen (Seitenzahlen, isolierte Überschriften, Leerseiten)
    all_chunks = [c for c in all_chunks if len(c.page_content.strip()) >= 30]

    print(f"✂️  {len(all_chunks)} Chunks erstellt")
    return all_chunks