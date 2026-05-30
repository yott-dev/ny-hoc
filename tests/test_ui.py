from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from src.agent import get_agent_response_stream

def test_get_agent_response_stream_returns_rag_context():
    fake_search = MagicMock()
    fake_search.invoke.return_value = "Fake RAG Context"

    fake_agent = MagicMock()
    fake_agent.stream.return_value = iter([])

    with patch("src.agent.get_cached_agent", return_value=(fake_agent, fake_search)):
        _, rag_context, _ = get_agent_response_stream("Frage?", MagicMock(), [])

    assert rag_context == "Fake RAG Context"

def test_get_agent_response_stream_collected_messages_starts_empty():
    fake_search = MagicMock()
    fake_search.invoke.return_value = "Fake RAG Context"

    fake_agent = MagicMock()
    fake_agent.stream.return_value = iter([])

    with patch("src.agent.get_cached_agent", return_value=(fake_agent, fake_search)):
        _, _, collected_messages = get_agent_response_stream("Frage?", MagicMock(), [])

    assert collected_messages == []

def test_get_agent_response_stream_yields_ai_chunks():
    fake_search = MagicMock()
    fake_search.invoke.return_value = "Fake RAG Context"

    fake_chunk = MagicMock(spec=AIMessageChunk)
    fake_chunk.content = "Antwort"

    fake_agent = MagicMock()
    fake_agent.stream.return_value = iter([
        (fake_chunk, {"langgraph_node": "model"})
    ])

    with patch("src.agent.get_cached_agent", return_value=(fake_agent, fake_search)):
        stream, _, _ = get_agent_response_stream("Frage?", MagicMock(), [])
        result = list(stream)  # Generator konsumieren

    assert result == ["Antwort"]