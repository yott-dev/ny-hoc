import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import re
import time


def slugify(text: str) -> str:
    """Converts text to a filename-safe slug."""
    replacements = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
                    'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[\s_-]+', '_', text)
    return text[:50].strip('_')


def extract_topic(link_text: str) -> str:
    """Extracts the short topic from the ESTV link text.

    Link text format: 'Kreisschreiben Nr. X: TITLE TITLE vom DATE (Direkte Bundessteuer)PDF...'
    The title appears twice consecutively without a separator.
    """
    colon_idx = link_text.find(':')
    if colon_idx == -1:
        return ""

    after_colon = link_text[colon_idx + 1:].strip()

    # Entferne den Anhang: "vom DATE (Direkte Bundessteuer)PDF..."
    after_colon = re.sub(r'\s+vom\s+\d{1,2}\.\d{1,2}\.\d{4}.*', '', after_colon)
    after_colon = re.sub(r'\s*\(Direkte Bundessteuer\).*', '', after_colon)
    after_colon = re.sub(r'PDF[\d\.,\s].*', '', after_colon)

    title = after_colon.strip()
    if not title:
        return ""

    # Der Titeltext wird auf der Seite verdoppelt — kürzestes Präfix finden das sich wiederholt
    for i in range(1, len(title) // 2 + 1):
        if title[:i] == title[i:i * 2]:
            return title[:i].strip()

    return title[:60].strip()  # Fallback: erste 60 Zeichen


def download_pdfs(url: str, output_folder: str = "downloads") -> None:
    """Lädt alle PDFs von einer Webseite herunter, überspringt bereits vorhandene Dateien."""

    # Ordner erstellen
    os.makedirs(output_folder, exist_ok=True)

    # Webseiten blocken Anfragen ohne Browser-User-Agent und gibt möglicherweise eine leere oder andere Seite zurück. Daher implementieren wir das:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Webseite laden und parsen
    print(f"Lade Seite: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # PDF-Links mit Linktext extrahieren (dict statt set, um Titel zu behalten)
    pdf_links = {}  # {full_url: topic_slug}
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if ".pdf" in href.lower():
            full_url = urljoin(url, href)
            if full_url not in pdf_links:
                topic = extract_topic(link.get_text())
                pdf_links[full_url] = slugify(topic) if topic else ""

    print(f"Gefunden: {len(pdf_links)} PDFs\n")

    # PDFs herunterladen
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
            time.sleep(0.3)  # kleine Pause, um den Server nicht zu überlasten
        except Exception as e:
            print(f"[{i}/{len(pdf_links)}] ✗ {filename}: {e}")


def main():
    url = "https://www.estv.admin.ch/de/kreisschreiben-direkten-bundessteuer"
    download_pdfs(url, output_folder="ressources/pdf/estv_kreisschreiben")


if __name__ == "__main__":
    main()
