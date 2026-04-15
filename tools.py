import requests
import html2text


def search_zstb(query):
    search_url = "https://www.zh.ch/bin/zhweb/search.json"
    params = {
        "q":
        "fq":
        "rows":
    }

response = requests.get(search_url, params=params)
result = response.json().get('results'), []

if not results:
        return "Keine Einträge gefunden."

    # 2. Den ersten Link auslesen
    page_url = "https://www.zh.ch" + results[0]['url']
    page_content = requests.get(page_url).text
    
    # 3. HTML in Markdown umwandeln (viel besser für LLMs!)
    h = html2text.HTML2Text()
    h.ignore_links = False
    markdown_text = h.handle(page_content)
    
    return markdown_text

# Beispielaufruf:
# print(search_zstb("Abzug Krankheitskosten"))