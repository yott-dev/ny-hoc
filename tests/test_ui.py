from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessageChunk
from src.agent import get_agent_response_stream


# ─── AGENT STREAMING & UI INTEGRATION TESTS ────────────────────────────────────

def test_get_agent_response_stream_returns_rag_context():
    """Validates that the UI retrieval context is correctly extracted

    from the decoupled local context retrieval function.
    """
    fake_agent = MagicMock()
    fake_agent.stream.return_value = iter([])

    # Patch get_cached_agent to return the single object, and patch the real retrieval pipeline
    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})):
        _, rag_context, _ = get_agent_response_stream("Frage?", MagicMock(), [])

    assert rag_context == "Fake RAG Context"


def test_get_agent_response_stream_collected_messages_starts_empty():
    """Ensures the accumulator list for response tracking initializes as empty."""
    fake_agent = MagicMock()
    fake_agent.stream.return_value = iter([])

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})):
        _, _, collected_messages = get_agent_response_stream("Frage?", MagicMock(), [])

    assert collected_messages == []


def test_get_agent_response_stream_yields_ai_chunks():
    """Validates that the streaming generator correctly unpacks LangGraph message tuples

    and yields the textual content to the UI frontend.
    """
    fake_chunk = MagicMock(spec=AIMessageChunk)
    fake_chunk.content = "Antwort"

    fake_agent = MagicMock()
    # Simulating the exact LangGraph stream chunk structure: (MessageChunk, Metadata)
    fake_agent.stream.return_value = iter([
        (fake_chunk, {"langgraph_node": "model"})
    ])

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})):
        stream, _, _ = get_agent_response_stream("Frage?", MagicMock(), [])
        result = list(stream)  # Materialize the generator stream

    assert result == ["Antwort"]