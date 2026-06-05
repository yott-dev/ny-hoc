from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
import pytest
from src.custom_agents import TopicGuardrailMiddleware

# ---- TopicGuardrailMiddleware ----

def test_topic_guardrail_allows_tax_question():
    fake_result = MagicMock()
    fake_result.content = "ja"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_result

    with patch("src.custom_agents.classifier_llm", fake_llm):
        middleware = TopicGuardrailMiddleware()
        state = {"messages": [HumanMessage(content="Was ist die Quellensteuer?")]}
        result = middleware.before_model(state, runtime=None)
        assert result is None  # kein Fehler

def test_topic_guardrail_blocks_non_tax_question():
    fake_result = MagicMock()
    fake_result.content = "nein"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_result

    with patch("src.custom_agents.classifier_llm", fake_llm):
        middleware = TopicGuardrailMiddleware()
        state = {"messages": [HumanMessage(content="Was ist das Wetter heute?")]}
        with pytest.raises(ValueError, match="Steuerfragen"):
            middleware.before_model(state, runtime=None)


def test_topic_guardrail_extracts_question_from_rag_context():
    fake_result = MagicMock()
    fake_result.content = "ja"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_result

    with patch("src.custom_agents.classifier_llm", fake_llm):
        middleware = TopicGuardrailMiddleware()
        state = {"messages": [HumanMessage(content="Lokale Dokumente:\nRAG...\n\nFrage: Quellensteuer?")]}
        middleware.before_model(state, runtime=None)

        called_prompt = fake_llm.invoke.call_args[0][0]
        assert "Quellensteuer?" in called_prompt
        assert "Lokale Dokumente" not in called_prompt

