from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.custom_agents import TopicGuardrailMiddleware


# Helper to simulate the agent state structure
def create_state(messages):
    return {"messages": messages}


def test_topic_guardrail_allows_tax_question():
    """Verifies that the guardrail proceeds when the LLM classifies as 'ja'."""
    middleware = TopicGuardrailMiddleware()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="ja")

    state = create_state([HumanMessage(content="Wie hoch ist die Quellensteuer?")])

    with patch("src.custom_agents.classifier_llm", mock_llm):
        # Should not raise exception
        middleware.before_model(state, None)

    # Assert LLM was called correctly
    assert mock_llm.invoke.called


def test_topic_guardrail_blocks_non_tax_question():
    """Verifies that the guardrail raises ValueError when the LLM classifies as 'nein'."""
    middleware = TopicGuardrailMiddleware()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="nein")

    state = create_state([HumanMessage(content="Erzähl mir einen Witz")])

    with patch("src.custom_agents.classifier_llm", mock_llm):
        with pytest.raises(ValueError, match="Ich bin nur für Steuerfragen zuständig"):
            middleware.before_model(state, None)


def test_topic_guardrail_extracts_question_from_rag_context():
    """Verifies that the guardrail identifies the core question in RAG context injection."""
    middleware = TopicGuardrailMiddleware()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="ja")

    # Simulate a RAG prompt structure
    input_text = "Context: ... \n\nFrage: Wie hoch ist die Quellensteuer?"
    state = create_state([HumanMessage(content=input_text)])

    with patch("src.custom_agents.classifier_llm", mock_llm):
        middleware.before_model(state, None)

    # Verify extraction: Check LLM input content contains the query inside tags
    captured_messages = mock_llm.invoke.call_args[0][0]
    extracted_contents = [msg.content for msg in captured_messages if isinstance(msg, HumanMessage)]

    assert any("<frage>Wie hoch ist die Quellensteuer?</frage>" in content for content in extracted_contents)


def test_topic_guardrail_includes_ai_context():
    """Verifies that previous AI context is passed to the classifier."""
    middleware = TopicGuardrailMiddleware()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="ja")

    state = create_state([
        AIMessage(content="Die Steuerprogression ist progressiv."),
        HumanMessage(content="Und was ist mit den Abzügen?")
    ])

    with patch("src.custom_agents.classifier_llm", mock_llm):
        middleware.before_model(state, None)

    captured_messages = mock_llm.invoke.call_args[0][0]
    # Check if context block is present in the HumanMessage sent to classifier
    assert any("Vorherige Antwort des Assistenten:" in msg.content for msg in captured_messages if
               isinstance(msg, HumanMessage))