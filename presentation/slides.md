---
theme: default
title: RAG-Chatbot für Züricher Steuerrecht
info: Hands-on Chatbot Projekt – OST FS26
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
toc: false
layout: cover
hideNavigation: true
background: https://images.unsplash.com/photo-1527095655060-4026c4af2b25?q=80&w=3540&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
---

# <b> RAG-Chatbot </b>
## Zürcher Steuerrecht

Hands-on Chatbot · OST FS26

<div class="abs-br m-6 text-sm opacity-60">Yannick Ott <br>Noah Debbabi</div>


---

## Das Problem

<br><br><br><br>


- Bei Steuerfragen findet man kaum schnell präzise Antworten (grosse Datenmenge)
- Bei einem LLM ohne RAG fehlen die Quellangaben
- Steuerrecht ist komplex und ändert sich laufend.



---

## Unser Ziel

<br><br>

**Ein Chatbot der Fragen zu Züricher Steuerrecht beantwortet.**
<br><br>



- **Quellenbasiert:** jede Antwort mit Dokumentenangabe
- **Transparent:** Seite, Abschnitt und URL sichtbar
- **Aktuell:** greift auf Daten von 2025 zu und live auf das Züricher Steuerbuch
- **Ohne Halluzinationen:** antwortet nur aus Kontext



---

## Gesamtarchitektur: Agentic RAG

<br><br><br><br>

```mermaid
flowchart LR
  subgraph RAG["einmaliges Setup"]
    direction LR
    PDFs["PDFs"] --> MD["Markdown"]
    HTML["HTML"] --> MD
    MD --split_documents--> VStore["create Vector Store\nwith ChromaDB"]
    VStore --> EMB["OpenAI Embedding\nModell"]
  end

  EMB -->|"Chunk + Vektor + Metadaten"| DB[("ChromaDB")]

  S((" ")) --> FRAGE["Frage"]
  FRAGE --> FILTER["Middleware (Filter)"]
  FILTER --ungefährlich (local_search)--> DB
  FILTER --gefährlich--> ANS
  DB -->|"Kontext"| AGENT["Agent"]
  AGENT --> DEC{"Reicht lokaler\nKontext?"}
  DEC --nein--> TOOL["Tool: search_steuerbuch"]
  TOOL --> LIVE["live Abfrage\nauf zh.ch (API)"]
  LIVE --> ANS{"Antwort"}
  DEC --ja--> DIRECT["antwortet direkt\naus RAG"]
  DIRECT --> ANS
  ANS --> E((" "))
```


---

## Von Rohdokumenten zu Embeddings

<br><br><br><br>

```mermaid
flowchart LR
    subgraph Quellen
        A[PDF\nKreisschreiben\nSteuergesetze]
        B[HTML\nMerkblätter\nWegleitungen]
    end

    A --> C[pymupdf4llm\n→ Markdown\n+ Seitenzahl]
    B --> D[BeautifulSoup\n+ html2text\n→ Markdown]

    C --> E[MarkdownHeader\nTextSplitter\nnach Abschnitten]
    D --> E

    E --> F[Recursive\nCharacterSplitter\n750 Zeichen / 150 Overlap]

    F --> G[OpenAI\ntext-embedding-3-large]

    G --> H[(ChromaDB\nCosine Similarity)]
```

---

## Warum zwei Splitting-Schritte?

<br>



**Pass 1 — MarkdownHeaderTextSplitter**
- Teilt nach `##` Überschriften auf (z.B. *A. Steuerpflicht*)
- Jeder Chunk bekommt den Abschnittsnamen als Metadatum
- Semantisch zusammengehöriger Inhalt bleibt zusammen


<br>


**Pass 2 — RecursiveCharacterTextSplitter**
- Maximal 750 Zeichen pro Chunk, 150 Überlappung
- Verhindert zu grosse Chunks die das Kontextfenster sprengen
- Quell-Metadaten (Datei, Seite, Abschnitt) bleiben erhalten




---


## Ergebnis
<br><br><br><br><br>
Quellenangabe zeigt für PDFs: 
<br>
`steuergesetz.pdf (Seite 3, A. Steuerpflicht)`
<br><br><br>
Quellenangabe zeigt für Steuberbuch Webseite: 
<br>
`https://www.zh.ch/de/steuern-finanzen/steuern/treuhaender/steuerbuch/steuerbuch-definition/zstb-3-1.html?search=miete`



---

## Warum ein Agent?

<br><br><br>
Eine klassische RAG-Kette kann nur **lokale Dokumente** befragen.
<br>
Der Agent kann **selbst entscheiden**:
- Reichen die lokalen Docs?
- Soll das Steuerbuch online abgefragt werden?

<br><br>

**RAG-first Strategie:**
RAG wird **immer zuerst** ausgeführt. Der Kontext wird als Pflicht-Basis an den Agent übergeben und er antwortet nie aus dem Gedächtnis.


---

## Halluzinations-Unterdrückung
<br>


### 1. RAG-first
Lokale Dokumente werden bei **jeder** Anfrage vorab geladen und als Kontext mitgegeben — unabhängig davon ob der Agent sie nutzt.

<br>

### 2. System-Prompt
```
"Nutze NUR den mitgegebenen Kontext als Wissensbasis und
ergänze NIEMALS aus dem Gedächtnis."
```
<br>

### 3. Expliziter Fallback
Wenn keine eindeutige Antwort möglich ist:

> *"Dazu konnte ich keine eindeutige Antwort finden. Bitte wenden Sie sich an das kantonale Steueramt Zürich."*




---

## Evaluation mit RAGAS

<br>

| Metrik | Beschreibung               | RAG  | Agentic RAG |
|---|----------------------------|------|-------------|
| **Answer Accuracy** | Korrektheit der Antwort    | 0.48 | 0.43        |
| **Faithfulness** | Antwortet nur aus Kontext? | 0.92 | 0.85        |
| **Answer Relevancy** | Beantwortet die Frage?     | 0.86 | 0.64          |
| **Context Precision** | Richtige Chunks gefunden?  | 0.74 | 0.71          |
| **Context Recall** | Alle relevanten Chunks?    | 0.84 | 0.59          |

---


## Kontrollpunkte (Security, Middleware, LLM, Input)
<br>

```mermaid

flowchart LR
  S((" ")) --> AUTH["Authentication"]
  AUTH --> FRAGE["Frage"]
  FRAGE --> FILTER["Middleware (Filter)"]
  FILTER --ungefährlich (local_search)--> DB
  DB --dynamisch--> ANS["Antwort"]
  FILTER --gefährlich--> COD["Code"]
  COD --statisch--> ANS["Antwort"]
  ANS --> E((" "))
```

| Kontrollsegmente   | Beschreibung                                   |
|--------------------|------------------------------------------------|
| **Authentication** | Login für den User                             |
| **LLM-as-a-Guard** | Middleware (Überprüfung prompt)                |
| **Output Control** | LLM-Spezifikation wie llm, tokens, temperature |
| **Langfuse**       | Überwachung der LLM calls                      |

---


## Herausforderungen
<br>

- Komplexität der einzelnen Komponenten verstehen
- Live Abruf von Daten mit Sortierung nach Relevanz
- Architektur und Datenstruktur (Aufteilung des Codes) definieren

<br><br>

## Zukunftspotential

<br>

- Answer Accuracy bei 0.48
- Konsistente Antworten zu bekommen (Aufruf vom Tool wird nur selten gemacht)
- Feature-Erweiterung: Login, UI Elemente, Latenz
- Optimierung Code-Struktur


---
layout: center
class: text-center
text: 14pt
---

# Demo

*Live-Demonstration*

<div class="grid grid-cols-2 gap-8 mt-8 text-left">

<div class="border rounded p-4">
<div class="font-bold mb-2">Frage 1 — nur lokale Docs</div>
<div class="text-sm opacity-70">"Ab wann bin ich in Zürich steuerpflichtig?"</div>
<div class="text-xs mt-2 opacity-50">→ zeigt Quellenangabe mit Seite + Abschnitt</div>
</div>

<div class="border rounded p-4">
<div class="font-bold mb-2">Frage 2 — Steuerbuch-Tool</div>
<div class="text-sm opacity-70">"Was gilt für die Besteuerung von Homeoffice-Abzügen?"</div>
<div class="text-xs mt-2 opacity-50">→ triggert search_steuerbuch, zeigt zh.ch URLs</div>
</div>

</div>

---
layout: cover
background: https://images.unsplash.com/photo-1527095655060-4026c4af2b25?q=80&w=3540&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
---

# Fragen?

<div class="mt-8 text-sm opacity-60">
RAG-Chatbot für Züricher Steuerrecht · OST FS26
</div>

<div class="abs-br m-6 text-sm opacity-60">Yannick Ott <br>Noah Debbabi</div>
