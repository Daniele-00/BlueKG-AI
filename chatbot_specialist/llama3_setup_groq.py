"""
Configurazione LLAMA con Groq (wrapper custom - zero dipendenze extra)
"""

import os
from dotenv import load_dotenv
from groq import Groq
from typing import Optional, List, Any

# LangChain base (dovrebbe essere installato)
try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
except ImportError:
    print("❌ ERROR: langchain non installato!")
    print("Esegui: pip install langchain")
    exit(1)

# Carica environment
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY non trovata nel .env!")

print("✅ Groq configurato con Llama\n")

# ========================================
# WRAPPER GROQ PER LANGCHAIN
# ========================================


class GroqLLM(LLM):
    """Wrapper Groq per LangChain - compatibile con tutte le versioni"""

    model: str = "llama-3.1-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1000
    groq_api_key: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Groq(api_key=self.groq_api_key or GROQ_API_KEY)

    @property
    def _llm_type(self) -> str:
        return "groq-llama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Chiama Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Errore Groq API: {e}")
            raise


# ========================================
# FUNZIONI PER CREARE MODELLI
# ========================================


def get_translator_llm():
    """Traduttore IT→EN con Llama 3.1 8B"""
    return GroqLLM(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=200,
        groq_api_key=GROQ_API_KEY,
    )


def get_router_llm():
    """Router specialist con Llama 3.1 70B"""
    return GroqLLM(
        model="llama-3.1-70b-versatile",
        temperature=0.1,
        max_tokens=100,
        groq_api_key=GROQ_API_KEY,
    )


def get_coder_llm():
    """Generatore Cypher con Llama 3.1 70B"""
    return GroqLLM(
        model="llama-3.1-70b-versatile",
        temperature=0.1,
        max_tokens=2000,
        groq_api_key=GROQ_API_KEY,
    )


def get_synthesizer_llm():
    """Sintetizzatore con Llama 3.1 70B"""
    return GroqLLM(
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        max_tokens=1000,
        groq_api_key=GROQ_API_KEY,
    )


# ========================================
# EXPORT
# ========================================

__all__ = [
    "GroqLLM",
    "get_translator_llm",
    "get_router_llm",
    "get_coder_llm",
    "get_synthesizer_llm",
]

if __name__ == "__main__":
    # Test completo
    print("🧪 Test Llama su Groq...\n")

    # Test 1: Translator
    print("1. Test Translator (Llama 8B)...")
    llm = get_translator_llm()
    result = llm("Translate to English: Qual è il fatturato totale?")
    print(f"   ✅ Risposta: {result}\n")

    # Test 2: Coder
    print("2. Test Coder (Llama 70B)...")
    llm = get_coder_llm()
    result = llm("Generate a simple Cypher query to find total revenue")
    print(f"   ✅ Risposta: {result[:100]}...\n")

    # Test 3: Con LangChain Chain
    print("3. Test con LangChain Chain...")
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["text"], template="Translate to English: {text}"
    )

    chain = LLMChain(llm=get_translator_llm(), prompt=prompt)
    result = chain.run(text="Buongiorno, come stai?")
    print(f"   ✅ Chain result: {result}\n")

    print("✅ TUTTI I TEST PASSATI!")
