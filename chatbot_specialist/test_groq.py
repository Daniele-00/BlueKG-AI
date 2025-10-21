#!/usr/bin/env python3
"""
Test rapido integrazione Groq/Llama nel chatbot
Esegui dopo aver fatto le modifiche
"""

import os
import sys
from dotenv import load_dotenv

# Carica .env
load_dotenv()

print("=" * 80)
print("🧪 TEST INTEGRAZIONE GROQ/LLAMA")
print("=" * 80)
print()

# =======================================================
# TEST 1: Verifica API Key
# =======================================================
print("1️⃣ Verifica API Key...")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("   ❌ GROQ_API_KEY non trovata nel .env!")
    print("   Aggiungi: GROQ_API_KEY=gsk_xxxxxxxxxxxxx")
    sys.exit(1)
else:
    print(f"   ✅ GROQ_API_KEY trovata: {GROQ_API_KEY[:10]}...")
print()

# =======================================================
# TEST 2: Import wrapper
# =======================================================
print("2️⃣ Test import groq_wrapper...")

try:
    from groqWrapper import ChatGroq

    print("   ✅ groq_wrapper importato correttamente")
except ImportError as e:
    print(f"   ❌ Errore import groq_wrapper: {e}")
    print("   Assicurati che groq_wrapper.py sia nella stessa cartella")
    sys.exit(1)
print()

# =======================================================
# TEST 3: Test Groq SDK
# =======================================================
print("3️⃣ Test Groq SDK...")

try:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    print("   ✅ Groq SDK funzionante")
except Exception as e:
    print(f"   ❌ Errore Groq SDK: {e}")
    print("   Installa: pip install groq")
    sys.exit(1)
print()

# =======================================================
# TEST 4: Test wrapper con API reale
# =======================================================
print("4️⃣ Test ChatGroq wrapper con API...")

try:
    from langchain_core.messages import HumanMessage

    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Usa 8B per test veloce
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=50,
    )

    result = llm.invoke([HumanMessage(content="Say 'test ok' in Italian")])

    print(f"   ✅ Risposta Llama: {result.content}")

except Exception as e:
    print(f"   ❌ Errore test API: {e}")
    sys.exit(1)
print()

# =======================================================
# TEST 5: Test con chatbot_specialist
# =======================================================
print("5️⃣ Test integrazione chatbot_specialist...")

try:
    # Prova a importare config
    from chatbot_specialist import config, create_llm_model

    print("   ✅ chatbot_specialist importato")

    # Verifica configurazione
    try:
        coder_model = config.models["agent_models"]["coder"]
        print(f"   ℹ️  Modello coder configurato: {coder_model}")

        if "groq" in coder_model:
            print("   ✅ Coder usa Groq/Llama!")
        else:
            print(f"   ⚠️  Coder NON usa Groq (usa: {coder_model})")
            print("   → Cambia config/models.yaml per usare llama-70b-groq")
    except Exception as e:
        print(f"   ⚠️  Config non caricata correttamente: {e}")

except ImportError as e:
    print(f"   ⚠️  chatbot_specialist non importato: {e}")
    print("   → Normale se non hai ancora fatto le modifiche al file")
print()

# =======================================================
# TEST 6: Test creazione modello
# =======================================================
print("6️⃣ Test creazione modello tramite create_llm_model...")

try:
    from chatbot_specialist import create_llm_model

    # Prova a creare modello coder
    coder_llm = create_llm_model("coder")
    print(f"   ✅ Modello coder creato: {type(coder_llm).__name__}")

    if "Groq" in type(coder_llm).__name__:
        print("   ✅ Sta usando ChatGroq!")
    else:
        print(f"   ⚠️  Non sta usando Groq (tipo: {type(coder_llm).__name__})")

except Exception as e:
    print(f"   ⚠️  Errore creazione modello: {e}")
    print("   → Verifica che hai aggiunto il blocco 'groq' in create_llm_model()")
print()

# =======================================================
# RIEPILOGO
# =======================================================
print("=" * 80)
print("📊 RIEPILOGO")
print("=" * 80)
print()
print("Se tutti i test sopra hanno ✅, sei pronto!")
print()
print("Prossimi passi:")
print("  1. Avvia il chatbot: python chatbot_specialist.py")
print("  2. Testa una query Cypher")
print("  3. Verifica velocità migliorata (2-4s vs 25-45s)")
print()
print("=" * 80)
