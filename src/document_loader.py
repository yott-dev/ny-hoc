import pymupdf4llm
from dotenv import load_dotenv
load_dotenv()
import html2text
from bs4 import BeautifulSoup
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from src.config import PDF_DIR, MD_DIR, HTML_DIR, CHUNK_OVERLAP, CHUNK_SIZE, TABLES_FOLDER_NAME, MIN_CHUNK_LENGTH


def _load_pdfs() -> list[Document]:
    """Parses all PDFs using pymupdf4llm while preserving page metadata."""
    docs: list[Document] = []
    if not PDF_DIR.exists():
        return docs

    for pdf_path in PDF_DIR.rglob("*.pdf"):
        pages: list[dict[str, Any]] = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        for i, page in enumerate(pages):
            meta: dict[str, Any] = page.get("metadata", {})
            page_num: int = meta.get("page", meta.get("page_number", i + 1))

            docs.append(Document(
                page_content=page["text"],
                metadata={
                    "source": pdf_path.name,
                    "page": page_num,
                    "folder": pdf_path.parent.name
                }
            ))
    return docs


def _load_htmls() -> list[Document]:
    """Parses HTML files, strips boilerplate components, and converts content to Markdown."""
    docs: list[Document] = []
    if not HTML_DIR.exists():
        return docs

    h2t_converter = html2text.HTML2Text()
    h2t_converter.ignore_links = False
    h2t_converter.body_width = 0

    for html_path in HTML_DIR.glob("*.html"):
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

        # Strip structural boilerplate components
        for tag in soup(["nav", "footer", "header", "script", "style"]):
            tag.decompose()

        main_content = soup.find("main") or soup.find("article") or soup.body
        md_text = h2t_converter.handle(str(main_content))

        docs.append(Document(
            page_content=md_text,
            metadata={"source": html_path.name}
        ))
    return docs


def _load_mds() -> list[Document]:
    """Reads native Markdown files directly from disk without format conversion."""
    docs: list[Document] = []
    if not MD_DIR.exists():
        return docs

    for md_path in MD_DIR.glob("*.md"):
        docs.append(Document(
            page_content=md_path.read_text(encoding="utf-8"),
            metadata={"source": md_path.name}
        ))
    return docs


def load_documents() -> list[Document]:
    """Ingests and normalizes PDFs, HTML, and Markdown files into a consolidated list of raw sections."""
    docs: list[Document] = []

    docs.extend(_load_pdfs())
    docs.extend(_load_htmls())
    docs.extend(_load_mds())

    print(f"📦 {len(docs)} document sections loaded.")
    return docs


def _deduplicate_chunks(chunks: list[Document]) -> list[Document]:
    """Removes structurally identical document chunks based on their source and content signature."""
    seen: set[str] = set()
    unique_chunks: list[Document] = []

    for chunk in chunks:
        key = f"{chunk.metadata.get('source', '')}|{chunk.page_content}"
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    removed = len(chunks) - len(unique_chunks)
    if removed:
        print(f"🔁 {removed} duplicate chunks removed.")
    return unique_chunks


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Executes a two-pass chunking strategy using Markdown headers and character constraints.
    Bypasses secondary splitting for structural tables to preserve row context and filters out noise.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False
    )

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks: list[Document] = []

    for doc in docs:
        header_chunks = md_splitter.split_text(doc.page_content)

        for chunk in header_chunks:
            chunk.metadata.update(doc.metadata)

        if doc.metadata.get("folder") == TABLES_FOLDER_NAME:
            all_chunks.extend(header_chunks)
        else:
            all_chunks.extend(char_splitter.split_documents(header_chunks))

    clean_chunks = [c for c in all_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]
    unique_chunks = _deduplicate_chunks(clean_chunks)

    print(f"✂️  {len(unique_chunks)} final unique chunks generated.")
    return unique_chunks