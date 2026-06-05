from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from pathlib import Path


BASE_RESOURCE_DIR = Path("ressources")
PDF_DIR = BASE_RESOURCE_DIR / "pdf"
HTML_DIR = BASE_RESOURCE_DIR / "html"
MD_DIR = BASE_RESOURCE_DIR / "md"

MAX_HISTORY_PAIRS = 6

CHUNK_SIZE = 750
CHUNK_OVERLAP = 250
MIN_CHUNK_LENGTH = 30
TABLES_FOLDER_NAME = "Tabellen"

TOP_K_RERANKED = 6
MULTI_QUERY_COUNT = 3
SIMILARITY_SEARCH_K = 5

BATCH_SIZE = 500

agent_llm = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    max_tokens=500,
    temperature=0,
    #for streaming
    streaming=True
)