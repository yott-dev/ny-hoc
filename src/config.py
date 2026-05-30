from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

MAX_HISTORY_PAIRS = 6


agent_llm = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    max_tokens=3000,
    temperature=0,
    #for streaming
    streaming=True
)