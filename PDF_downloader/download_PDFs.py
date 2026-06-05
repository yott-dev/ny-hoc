import os
import re
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup


def slugify(text: str) -> str:
    """
    Converts raw text into a filename-safe string by replacing German umlauts
    and stripping away illegal punctuation marks.
    """
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
    }
    for key, value in replacements.items():
        text = text.replace(key, value)

    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[\s_-]+', '_', text)
    return text[:50].strip('_')


def extract_topic(link_text: str) -> str:
    """
    Extracts the cleaned topic title from the ESTV circular link string.

    Expected string layout pattern:
    'Kreisschreiben Nr. X: TITLE TITLE vom DATE (Direkte Bundessteuer)PDF...'
    Note: The source website repeats the core title twice without separators.
    """
    colon_idx = link_text.find(':')
    if colon_idx == -1:
        return ""

    after_colon = link_text[colon_idx + 1:].strip()

    # Strip structural trailing tags: "vom DATE (Direkte Bundessteuer)PDF..."
    after_colon = re.sub(r'\s+vom\s+\d{1,2}\.\d{1,2}\.\d{4}.*', '', after_colon)
    after_colon = re.sub(r'\s*\(Direkte Bundessteuer\).*', '', after_colon)
    after_colon = re.sub(r'PDF[\d\.,\s].*', '', after_colon)

    title = after_colon.strip()
    if not title:
        return ""

    # Deduplicate the mirrored prefix string anomaly caused by the website's HTML layout
    for i in range(1, len(title) // 2 + 1):
        if title[:i] == title[i:i * 2]:
            return title[:i].strip()

    return title[:60].strip()  # Fallback boundary limit


def download_pdfs(url: str, output_folder: str = "downloads") -> None:
    """
    Scrapes and saves PDF files from the target ESTV document library.

    Skips redundant asset downloads if the calculated file names or raw filenames
    already exist within the output destination folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Web servers routinely block bare connections lacking browser headers
    # to safeguard against Denial of Service (DoS) scraping floods.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    print(f"Lade Seite: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Map tracking entries to isolate duplicates while preserving specific topic strings
    pdf_links: dict[str, str] = {}  # Format sequence: {full_url: topic_slug}
    for link in soup.find_all("a", href=True):
        href: str = link["href"]
        if ".pdf" in href.lower():
            full_url = urljoin(url, href)
            if full_url not in pdf_links:
                topic = extract_topic(link.get_text())
                pdf_links[full_url] = slugify(topic) if topic else ""

    print(f"Gefunden: {len(pdf_links)} PDFs\n")

    # Execute binary stream downloads sequentially
    for i, (pdf_url, topic_slug) in enumerate(pdf_links.items(), 1):
        base_name = pdf_url.split("/")[-1].split("?")[0]
        stem, ext = os.path.splitext(base_name)
        filename = f"{stem}_{topic_slug}{ext}" if topic_slug else base_name
        filepath = os.path.join(output_folder, filename)

        original_filepath = os.path.join(output_folder, base_name)
        if os.path.exists(filepath) or os.path.exists(original_filepath):
            print(f"[{i}/{len(pdf_links)}] ⏭  {filename} (bereits vorhanden)")
            continue

        try:
            pdf_response = requests.get(pdf_url, headers=headers)
            pdf_response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(pdf_response.content)

            print(f"[{i}/{len(pdf_links)}] ✓ {filename}")

            # Rate limiting delay block to comply with polite crawling policies
            time.sleep(0.3)
        except Exception as error:
            print(f"[{i}/{len(pdf_links)}] ✗ {filename}: {error}")


def main() -> None:
    """
    Main application entrance script driving target document extraction pipelines.
    """
    url = "https://www.estv.admin.ch/de/kreisschreiben-direkten-bundessteuer"
    download_pdfs(url, output_folder="ressources/pdf/estv_kreisschreiben")


if __name__ == "__main__":
    main()