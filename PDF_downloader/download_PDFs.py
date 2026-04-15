import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time


def download_pdfs(url: str, output_folder: str = "downloads") -> None:
    """Lädt alle PDFs von einer Webseite herunter"""

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


    # PDF-Links extrahieren/sammeln
    pdf_links = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if ".pdf" in href.lower():
            full_url = urljoin(url, href)
            pdf_links.add(full_url)
    
    print(f"Gefunden: {len(pdf_links)} PDFs\n")
    
    # PDFs herunterladen
    for i, pdf_url in enumerate(pdf_links, 1):
        filename = pdf_url.split("/")[-1].split("?")[0]
        filepath = os.path.join(output_folder, filename)
        
        try:
            pdf_response = requests.get(pdf_url, headers=headers)
            pdf_response.raise_for_status()
            
            with open(filepath, "wb") as f:
                f.write(pdf_response.content)
            
            print(f"[{i}/{len(pdf_links)}] ✓ {filename}")
        except Exception as e:
            print(f"[{i}/{len(pdf_links)}] ✗ {filename}: {e}")


def main():
    url = "https://www.estv.admin.ch/de/kreisschreiben-direkten-bundessteuer"
    download_pdfs(url, output_folder="estv_kreisschreiben")


if __name__ == "__main__":
    main()