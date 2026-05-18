"""
Chunk-Inspektion: Zeigt Statistiken und Beispiele pro Dokumenttyp.
Ausführen: uv run python evals/inspect_chunks.py
"""
import sys
import pymupdf4llm
import html2text
from pathlib import Path
from bs4 import BeautifulSoup
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# ── gleiche Konfiguration wie main.py ──────────────────────────────────────
def load_documents() -> list[Document]:
    docs = []

    for pdf_path in Path("ressources/pdf").rglob("*.pdf"):
        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        for i, page in enumerate(pages):
            meta = page.get("metadata", {})
            page_num = meta.get("page", meta.get("page_number", i + 1))
            docs.append(Document(
                page_content=page["text"],
                metadata={"source": pdf_path.name, "page": page_num, "folder": pdf_path.parent.name}
            ))

    _h2t = html2text.HTML2Text()
    _h2t.ignore_links = False
    _h2t.body_width = 0
    for html_path in Path("ressources/html").glob("*.html"):
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        for tag in soup(["nav", "footer", "header", "script", "style"]):
            tag.decompose()
        main_content = soup.find("main") or soup.find("article") or soup.body
        md_text = _h2t.handle(str(main_content))
        docs.append(Document(page_content=md_text, metadata={"source": html_path.name}))

    for md_path in Path("ressources/md").glob("*.md"):
        docs.append(Document(
            page_content=md_path.read_text(encoding="utf-8"),
            metadata={"source": md_path.name}
        ))

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    all_chunks = []
    for doc in docs:
        header_chunks = md_splitter.split_text(doc.page_content)
        for chunk in header_chunks:
            chunk.metadata.update(doc.metadata)
        if doc.metadata.get("folder") == "Tabellen":
            all_chunks.extend(header_chunks)
        else:
            all_chunks.extend(char_splitter.split_documents(header_chunks))
    return all_chunks


# ── Analyse ─────────────────────────────────────────────────────────────────
def analyse(chunks: list[Document], sample_n: int = 2):
    # Gruppieren nach Quelle (source)
    by_source: dict[str, list[Document]] = defaultdict(list)
    for c in chunks:
        by_source[c.metadata.get("source", "?")].append(c)

    print(f"\n{'='*70}")
    print(f"TOTAL: {len(chunks)} Chunks aus {len(by_source)} Quellen")
    print(f"{'='*70}\n")

    # Übersichtstabelle
    print(f"{'Quelle':<55} {'Chunks':>6} {'Min':>5} {'Avg':>5} {'Max':>5}  Probleme")
    print("-" * 90)

    problems_total = []
    for source, cs in sorted(by_source.items()):
        lengths = [len(c.page_content) for c in cs]
        avg = sum(lengths) // len(lengths)
        min_l, max_l = min(lengths), max(lengths)

        # Probleme erkennen
        problems = []
        empty = [c for c in cs if len(c.page_content.strip()) < 30]
        huge  = [c for c in cs if len(c.page_content) > 1200]
        if empty:
            problems.append(f"{len(empty)} leer/zu kurz")
        if huge:
            problems.append(f"{len(huge)} zu gross (>{1200})")

        problems_total.extend(empty)
        flag = "⚠️  " + ", ".join(problems) if problems else "✅"
        print(f"{source:<55} {len(cs):>6} {min_l:>5} {avg:>5} {max_l:>5}  {flag}")

    # Stichproben pro Quelle
    if "--samples" in sys.argv:
        print(f"\n{'='*70}")
        print("STICHPROBEN (erste und letzte Chunks pro Quelle)")
        print(f"{'='*70}")
        for source, cs in sorted(by_source.items()):
            print(f"\n── {source} ({len(cs)} Chunks) ──")
            for c in cs[:sample_n]:
                _print_chunk(c)
            if len(cs) > sample_n:
                print("  ...")
                _print_chunk(cs[-1])

    # Problematische Chunks im Detail
    if problems_total:
        print(f"\n{'='*70}")
        print(f"PROBLEMATISCHE CHUNKS ({len(problems_total)} gesamt)")
        print(f"{'='*70}")
        for c in problems_total[:10]:
            _print_chunk(c)


def _print_chunk(c: Document):
    meta = c.metadata
    source = meta.get("source", "?")
    page   = meta.get("page", "")
    folder = meta.get("folder", "")
    h2     = meta.get("h2", meta.get("h1", ""))
    label  = f"{source}" + (f" S.{page}" if page else "") + (f" [{h2}]" if h2 else "")
    content = c.page_content.strip().replace("\n", " ")
    print(f"\n  [{label}] {len(c.page_content)} Zeichen")
    print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")


if __name__ == "__main__":
    print("Lade Dokumente...")
    docs = load_documents()
    print(f"{len(docs)} Seiten/Dokumente geladen")

    print("Splitte Chunks...")
    chunks = split_documents(docs)

    analyse(chunks)

    print("\nTipp: Mit --samples auch Chunk-Inhalte anzeigen:")
    print("  uv run python evals/inspect_chunks.py --samples")
