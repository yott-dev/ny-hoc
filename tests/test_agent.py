from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessageChunk
from src.agent import get_agent_response, get_agent_response_stream


# ─── MOCKS FOR AGENT INVOKE (NON-STREAM) TESTS ──────────────────────────────────

def test_get_agent_response_returns_rag_context():
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": []}

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})), \
            patch("src.agent.get_langfuse_handler", return_value=MagicMock()):
        result, rag_context, _ = get_agent_response("Was ist Quellensteuer?", MagicMock(), [])

    assert rag_context == "Fake RAG Context"


def test_get_agent_response_builds_messages_correctly():
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": []}

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})), \
            patch("src.agent.get_langfuse_handler", return_value=MagicMock()):
        get_agent_response("Was ist Quellensteuer?", MagicMock(), [])

    called_messages = fake_agent.invoke.call_args[0][0]["messages"]
    assert len(called_messages) == 1
    assert "Quellensteuer" in called_messages[0].content


def test_get_agent_response_with_chat_history():
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": []}

    history = [HumanMessage(content="Vorherige Frage")]

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})), \
            patch("src.agent.get_langfuse_handler", return_value=MagicMock()):
        get_agent_response("Neue Frage", MagicMock(), history)

    called_messages = fake_agent.invoke.call_args[0][0]["messages"]
    assert len(called_messages) == 2


# ─── MOCKS FOR AGENT STREAM TESTS ──────────────────────────────────────────────

def test_get_agent_response_stream_returns_rag_context():
    fake_agent = MagicMock()
    fake_agent.stream.return_value = iter([])

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})):
        _, rag_context, _ = get_agent_response_stream("Frage?", MagicMock(), [])

    assert rag_context == "Fake RAG Context"


def test_get_agent_response_stream_collected_messages_starts_empty():
    fake_agent = MagicMock()
    fake_agent.stream.return_value = iter([])

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})):
        _, _, collected_messages = get_agent_response_stream("Frage?", MagicMock(), [])

    assert collected_messages == []


def test_get_agent_response_stream_yields_ai_chunks():
    fake_chunk = MagicMock(spec=AIMessageChunk)
    fake_chunk.content = "Antwort"

    fake_agent = MagicMock()
    # Mocking the LangGraph chunk tuple structure: (message_chunk, metadata)
    fake_agent.stream.return_value = iter([
        (fake_chunk, {"langgraph_node": "model"})
    ])

    with patch("src.agent.get_cached_agent", return_value=fake_agent), \
            patch("src.agent.retrieve_local_context", return_value=("Fake RAG Context", {})):
        stream, _, _ = get_agent_response_stream("Frage?", MagicMock(), [])
        result = list(stream)

    assert result == ["Antwort"]