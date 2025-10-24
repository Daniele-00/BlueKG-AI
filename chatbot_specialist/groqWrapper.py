"""
Wrapper Groq per LangChain - compatibile con chatbot_specialist.py
Versione corretta per Pydantic v2
"""

from groq import Groq
from typing import Optional, List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr


class ChatGroq(BaseChatModel):
    """
    Wrapper Groq per LangChain

    Compatibile con sistema config YAML del chatbot.
    Supporta tutti i modelli Llama su Groq.
    """

    model: str = "llama-3.1-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1000
    groq_api_key: str = ""
    timeout: int = 30

    # Attributo privato per Pydantic v2
    _client: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Inizializza client dopo validazione Pydantic"""
        super().model_post_init(__context)
        if not self.groq_api_key:
            raise ValueError("groq_api_key è obbligatorio")
        self._client = Groq(api_key=self.groq_api_key)

    @property
    def _llm_type(self) -> str:
        return "groq-llama"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Genera risposta tramite Groq API"""

        # Converti messaggi LangChain in formato Groq
        groq_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                groq_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                groq_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                groq_messages.append({"role": "assistant", "content": msg.content})

        try:
            # Chiama API Groq usando self._client
            response = self._client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )

            content = response.choices[0].message.content
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"❌ Errore Groq API ({self.model}): {e}")
            raise


# Test del wrapper
if __name__ == "__main__":
    import os
    from langchain_core.messages import HumanMessage

    print("🧪 Test ChatGroq wrapper...\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY non trovata nel .env")
        print("   Aggiungi: GROQ_API_KEY=gsk_xxxxxxxxxxxxx")
        exit(1)

    # Test 1: Llama 70B
    print("1. Test Llama 3.1 70B...")
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        groq_api_key=api_key,
        temperature=0.1,
        max_tokens=200,
    )

    result = llm.invoke(
        [HumanMessage(content="Translate to English: Qual è il fatturato totale?")]
    )
    print(f"   Risposta: {result.content}\n")

    # Test 2: Llama 8B
    print("2. Test Llama 3.1 8B (instant)...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=100,
    )

    result = llm.invoke([HumanMessage(content="Say 'Hello' in Italian")])
    print(f"   Risposta: {result.content}\n")

    print("Wrapper funziona perfettamente!")
