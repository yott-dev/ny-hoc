"""
Evaluation-Runner — ohne Streamlit.
Verwendung:
  uv run python evals/run_eval.py          # Agent-Evaluation (Standard)
  uv run python evals/run_eval.py --rag    # Einfache RAG-Chain Evaluation
  uv run python evals/run_eval.py --plot   # Nur Diagramm anzeigen
"""
import sys
from pathlib import Path

# Projekt-Root zum Suchpfad hinzufügen damit src/ gefunden wird
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import sys
from src.vector_store import create_vector_store_with_embeddings
from src.agent import get_agent_response
from evals.ragas_eval import run_evaluation, run_agent_evaluation, plot_experiments

if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot_experiments(base="steuerbot_agent", save=True)
    elif "--rag" in sys.argv:
        vector_store = create_vector_store_with_embeddings()
        run_evaluation(vector_store)
    else:
        vector_store = create_vector_store_with_embeddings()
        run_agent_evaluation(vector_store, get_agent_response)
