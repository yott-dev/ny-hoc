from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

classifier_llm = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    max_tokens=3,
    temperature=0
)

class TopicGuardrailMiddleware(AgentMiddleware):
    @property
    def name(self) -> str:
        return "TopicGuardrail"

    def before_model(self, state, runtime):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                result = classifier_llm.invoke(
                    f"""Ist diese Frage eine Steuerfrage? Antworte NUR mit 'ja' oder 'nein'.
                    Frage: {msg.content}"""
                )
                if "nein" in result.content.lower():
                    raise ValueError(
                        "Ich bin nur für Steuerfragen zuständig. "
                        "Bitte stellen Sie eine steuerbezogene Frage."
                    )
                break
        return None

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


class InformationGuardrailMiddleware(AgentMiddleware):
    @property
    def name(self) -> str:
        return "InformationGuardrail"

    def before_model(self, state, runtime):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                result = classifier_llm.invoke(
                    f"""Beinhaltet diese Frage persönliche Angaben? Antworte NUR mit 'ja' oder 'nein'.
                    Frage: {msg.content}"""
                )
                if "nein" in result.content.lower():
                    raise ValueError(
                        "Bitte gib keine persönlichen Anganben zu deiner Person preis"
                    )
                break
        return None

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)