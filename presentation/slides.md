---
theme: default
colorSchema: light
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
mermaid:
  theme: test
  themeVariables:
    primaryColor: '#f0fdf4'
    primaryTextColor: '#064e3b'
    primaryBorderColor: '#059669'
    lineColor: '#059669'
    secondaryColor: '#ecfdf5'
    tertiaryColor: '#f0fdf4'
    clusterBkg: '#f0fdf4'
    clusterBorder: '#059669'
    edgeLabelBackground: '#ffffff'
    nodeBorder: '#059669'
    nodeTextColor: '#064e3b'
    mainBkg: '#f0fdf4'
    fontFamily: 'ui-sans-serif, system-ui, sans-serif'
    fontSize: '15px'
---

# <b> Agentic-RAG-Chatbot </b>
## Zürcher Steuerrecht

Hands-on Chatbot · OST FS26

<div class="abs-br m-6 text-sm opacity-60">Yannick Ott <br>Noah Debbabi</div>


---

## <b> <h1 class="text-emerald-600">Das Problem </h1></b>

<br><br><br>




- Bei Steuerfragen findet man kaum schnell präzise Antworten (grosse Datenmenge)
- Bei einem LLM ohne RAG fehlen die Quellangaben
- Steuerrecht ist komplex und ändert sich laufend


---

## <b> <h1 class="text-emerald-600">Unser Ziel</h1> </b>

<br><br>

**Ein Chatbot der Fragen zu zürcher Steuerrecht beantwortet**
<br><br>


- **Quellenbasiert:** jede Antwort mit Dokumentenangabe
- **Transparent:** Seite, Abschnitt und URL sichtbar
- **Aktuell:** greift auf Daten von 2025 zu und live auf das Züricher Steuerbuch
- **Ohne Halluzinationen:** antwortet nur aus Kontext



---

## <b> <h1 class="text-emerald-600"> Gesamtarchitektur: Agentic RAG </h1></b>

<br><br><br><br>

```mermaid {theme: 'dark'}
flowchart LR
  subgraph RAG["einmaliges Setup"]
    direction LR
    PDFs["PDFs"] --> MD["Markdown"]
    HTML["HTML"] --> MD
    MD --split_documents--> VStore["create Vector Store\nwith ChromaDB"]
    VStore --> EMB["OpenAI Embedding\nModell"]
  end

  EMB -->|"Chunk + Vektor + Metadaten"| DB[("ChromaDB")]

  S((" ")) --> AUTHENTICATION[Authentication]
  AUTHENTICATION --> FRAGE["Frage"]
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

## <b> <h1 class="text-emerald-600">Von Rohdokumenten zu Embeddings</h1> </b>

<br><br><br><br>

```mermaid {theme: 'dark'}
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

## <b> <h1 class="text-emerald-600">Warum zwei Splitting-Schritte?</h1> </b>

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


## <b> <h1 class="text-emerald-600"> Ergebnis </h1> </b>

<br>
```python
METADATEN: {'h2': '**A. Grundlagen**', 'page': 1, 'source': 'Merkbl_Selbst_ZH_2025.pdf'}
INHALT: . 2 StG). Darüber hinaus ist mit der Steuererklärung das Hilfsblatt A einzureichen, das einen...
```
<br><br>
Quellenangabe zeigt für PDFs: 
<br>
```
steuergesetz.pdf (Seite 3, A. Steuerpflicht)
```
<br><br>
Quellenangabe zeigt für Steuerbuch Webseite: 

<br>
```
https://www.zh.ch/de/steuern-finanzen/steuern/treuhaender/steuerbuch/steuerbuch-definition/zstb-3-1.html?search=miete
```



---

## <b> <h1 class="text-emerald-600"> Warum ein Agent? </h1> </b>

<br><br><br>
Eine klassische RAG-Kette kann nur lokale Dokumente befragen.
<br>
Der Agent kann **selbst entscheiden**:
- Genügen die lokalen Docs?
- Soll das Steuerbuch online abgefragt werden?

<br><br>

**RAG-first Strategie:**
RAG wird **immer zuerst** ausgeführt. Der Kontext wird als Pflicht-Basis an den Agent übergeben und er antwortet nie aus dem Gedächtnis.


---

## <b> <h1 class="text-emerald-600"> Halluzinations-Unterdrückung </h1> </b>
<br>


### 1. RAG-first
Lokale Dokumente werden bei jeder Anfrage vorab geladen und als Kontext mitgegeben, unabhängig davon ob der Agent sie nutzt.

<br>

### 2. System-Prompt
```
"Nutze NUR den mitgegebenen Kontext als Wissensbasis und ergänze NIEMALS aus dem Gedächtnis."
```

<br>

### 3. Expliziter Fallback 
Wenn keine eindeutige Antwort möglich ist:
```
"Dazu konnte ich keine eindeutige Antwort finden. Bitte wenden Sie sich an das kantonale Steueramt Zürich."
```



---

## <b> <h1 class="text-emerald-600"> Evaluation mit RAGAS </h1> </b> 

<br>

| Metrik | Beschreibung               | RAG  | Agentic RAG |
|---|----------------------------|------|------|
| **Answer Accuracy** | Korrektheit der Antwort    | 0.48 | 0.43 |
| **Faithfulness** | Antwortet nur aus Kontext? | 0.92 | 0.85 |
| **Answer Relevancy** | Beantwortet die Frage?     | 0.86 | 0.64 |
| **Context Precision** | Richtige Chunks gefunden?  | 0.74 | 0.71 |
| **Context Recall** | Alle relevanten Chunks?    | 0.84 | 0.59 |

---


## <b> <h1 class="text-emerald-600"> Kontrollpunkte (Security, Middleware, LLM, Input) </h1> </b>
<br>

```mermaid {theme: 'dark'}

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

| Kontrollsegmente   | Beschreibung                             |
|--------------------|------------------------------------------|
| **Authentication** | Login für den User                       |
| **LLM-as-a-Guard** | Middleware (Überprüfung Prompt)          |
| **Output Control** | LLM-Spezifikation wie llm und max_tokens |
| **Langfuse**       | Überwachung der LLM calls                |

---

# <b> <h1 class="text-emerald-600"> Live-Demo </h1> </b>

<br>

<div class="grid grid-cols-2 gap-8 mt-8 text-left">

<div class="border rounded p-4">
<div class="font-bold mb-2">Frage 1 — nur lokale Docs</div>
<div class="text-sm opacity-70">"Muss ich meinen Lottogewinn versteurn?"</div>
<div class="text-xs mt-2 opacity-50">→ zeigt Quellenangabe mit Seite + Abschnitt</div>
</div>

<div class="border rounded p-4">
<div class="font-bold mb-2">Frage 2 — Steuerbuch-Tool</div>
<div class="text-sm opacity-70">"Kann ich meine Miete von den Steuern abziehen?"</div>
<div class="text-xs mt-2 opacity-50">→ triggert search_steuerbuch, zeigt zh.ch URLs</div>
</div>

</div>

---
