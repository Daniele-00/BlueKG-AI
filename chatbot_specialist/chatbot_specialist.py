from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_neo4j import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from cachetools import TTLCache
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel
import re
import logging
import time
import json
from datetime import date
from neo4j.time import Date, DateTime, Time, Duration


# Importazione dei modelli LLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica .env
load_dotenv()


class Config:
    """Singleton per caricare tutte le config"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_all()
        return cls._instance

    def _load_yaml(self, filename: str) -> dict:
        """Carica un file YAML dalla cartella config/"""
        config_path = Path("config") / filename
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file non trovato: {config_path}")
            return {}
        except Exception as e:
            print(f"Errore caricamento {filename}: {e}")
            return {}

    def _load_all(self):
        """Carica tutti i file config"""
        self.models = self._load_yaml("models.yaml")
        self.database = self._load_yaml("database.yaml")
        self.system = self._load_yaml("system.yaml")
        self.fuzzy = self._load_yaml("fuzzy.yaml")

    def get_secret(self, key: str) -> str:
        """Ottieni un secret da .env"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Secret '{key}' non trovato in .env")
        return value

    def get_model_config(self, agent_name: str) -> dict:
        """Ottieni config completa per un agente"""
        model_id = self.models["agent_models"][agent_name]
        model_def = self.models["available_models"][model_id]
        provider = self.models["providers"][model_def["provider"]]

        return {
            "model_id": model_id,
            "model_name": model_def["model_name"],
            "provider": model_def["provider"],
            "provider_config": provider,
        }


# Istanza globale
config = Config()


# Funzione helper per creare modelli
def create_llm_model(agent_name: str):
    """Crea un'istanza LLM basata sulla config"""
    cfg = config.get_model_config(agent_name)

    if cfg["provider"] == "openai":
        api_key = config.get_secret(cfg["provider_config"]["api_key_env"])
        return ChatOpenAI(
            model=cfg["model_name"],
            temperature=cfg["provider_config"]["temperature"],
            api_key=api_key,
            timeout=cfg["provider_config"]["timeout"],
        )

    elif cfg["provider"] == "ollama":
        return ChatOllama(
            model=cfg["model_name"],
            temperature=0,
            base_url=cfg["provider_config"]["base_url"],
            request_timeout=cfg["provider_config"]["timeout"],
        )

    else:
        raise ValueError(f"Provider sconosciuto: {cfg['provider']}")


# Creazione modelli stratificati
MODELLI_STRATIFICATI = {
    agent: create_llm_model(agent)
    for agent in ["contextualizer", "router", "coder", "synthesizer", "translator"]
}

print(f"Modelli caricati da config:")
for agent, model in MODELLI_STRATIFICATI.items():
    print(f"   - {agent}: {config.models['agent_models'][agent]}")


# CONFIGURAZIONE LOGGING (DA CONFIG)
logging.basicConfig(
    level=getattr(logging, config.system["logging"]["level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURAZIONE FASTAPI E CACHE (DA CONFIG)
app = FastAPI()

# Cache per query Cypher (parametri da config/system.yaml)
query_cache = TTLCache(
    maxsize=config.system["cache"]["max_cache_size"],
    ttl=config.system["cache"]["query_ttl_seconds"],
)

# Timeout sessioni (da config/system.yaml)
SESSION_TIMEOUT = timedelta(minutes=config.system["sessions"]["timeout_minutes"])

# Limiti query (da config/system.yaml)
MAX_RESULTS_FOR_SYNTHESIZER = config.system["query_limits"][
    "max_results_for_synthesizer"
]
MAX_MESSAGES_PER_SESSION = config.system["sessions"]["max_messages_per_session"]

logger.info(
    f"Cache configurata: max_size={config.system['cache']['max_cache_size']}, ttl={config.system['cache']['query_ttl_seconds']}s"
)
logger.info(f"Session timeout: {config.system['sessions']['timeout_minutes']} minuti")
logger.info(f"Max risultati per synthesizer: {MAX_RESULTS_FOR_SYNTHESIZER}")


# CONNESSIONE NEO4J (DA CONFIG + SECRETS DA .ENV)
try:
    graph = Neo4jGraph(
        url=config.database["neo4j"]["url"],
        username=config.database["neo4j"]["username"],
        password=config.get_secret(config.database["neo4j"]["password_env"]),
    )
    logger.info(f"Connesso a Neo4j: {config.database['neo4j']['url']}")
except Exception as e:
    logger.error(f"Errore connessione Neo4j: {e}")
    raise


# Definizione degli schemi per le richieste e le risposte
class ConversationMessage(BaseModel):
    timestamp: datetime
    question: str
    answer: str
    context: List[Dict] = []
    query_generated: str = ""


class ConversationSession(BaseModel):
    user_id: str
    messages: List[ConversationMessage] = []
    created_at: datetime
    last_activity: datetime


# Cache globale per le sessioni di conversazione (scade dopo 30 minuti di inattività)
conversation_sessions: Dict[str, ConversationSession] = {}
SESSION_TIMEOUT = timedelta(minutes=config.system["sessions"]["timeout_minutes"])


def get_or_create_session(user_id: str) -> ConversationSession:
    """Ottieni o crea una sessione di conversazione per l'utente"""
    now = datetime.now()

    # Pulisci sessioni scadute
    expired_sessions = [
        uid
        for uid, session in conversation_sessions.items()
        if now - session.last_activity > SESSION_TIMEOUT
    ]
    for uid in expired_sessions:
        del conversation_sessions[uid]
        logger.info(f" Sessione scaduta eliminata per utente: {uid}")

    # Ottieni o crea sessione
    if user_id not in conversation_sessions:
        conversation_sessions[user_id] = ConversationSession(
            user_id=user_id, created_at=now, last_activity=now
        )
        logger.info(f" Nuova sessione creata per utente: {user_id}")
    else:
        conversation_sessions[user_id].last_activity = now

    return conversation_sessions[user_id]


def add_message_to_session(
    user_id: str, question: str, answer: str, context: List[Dict], query: str
):
    """Aggiungi un messaggio alla sessione di conversazione"""
    session = get_or_create_session(user_id)

    message = ConversationMessage(
        timestamp=datetime.now(),
        question=question,
        answer=answer,
        context=context,
        query_generated=query,
    )

    session.messages.append(message)

    # Mantengo solo gli ultimi 5 messaggi per evitare memory overflow
    if len(session.messages) > MAX_MESSAGES_PER_SESSION:
        session.messages = session.messages[-MAX_MESSAGES_PER_SESSION:]

    logger.info(
        f" Messaggio aggiunto alla sessione. Totale messaggi: {len(session.messages)}"
    )


def get_conversation_context(user_id: str) -> str:
    """Genera contesto conversazione COMPATTO"""
    session = conversation_sessions.get(user_id)
    if not session or len(session.messages) == 0:
        return ""

    last_msg = session.messages[-1]

    context_lines = ["CONVERSAZIONE PRECEDENTE (ultimo messaggio):"]
    context_lines.append(f"Domanda: {last_msg.question}")

    # Estrai SOLO nomi, niente altro
    if last_msg.context:
        nomi_clienti = []
        nomi_fornitori = []

        for item in last_msg.context:
            if "cliente" in item and item["cliente"]:
                nomi_clienti.append(item["cliente"])
            elif "fornitore" in item and item["fornitore"]:
                nomi_fornitori.append(item["fornitore"])

        if nomi_clienti:
            # Lista Python per facilitare parsing LLM
            context_lines.append(f"LISTA_CLIENTI = {nomi_clienti}")
        if nomi_fornitori:
            context_lines.append(f"LISTA_FORNITORI = {nomi_fornitori}")

    return "\n".join(context_lines)


def correggi_nomi_fuzzy_neo4j(question: str) -> tuple[str, list]:
    """Usa full-text search Neo4j per correggere typo nei nomi di entità."""

    # CHECK: Fuzzy abilitato?
    if not config.fuzzy["fuzzy_matching"]["enabled"]:
        logger.debug("🔇 Fuzzy matching disabilitato da config")
        return question, []

    correzioni = []

    # PARAMETRI DA CONFIG
    BLACKLIST = set(config.fuzzy["fuzzy_matching"]["blacklist"])
    MIN_WORD_LENGTH = config.fuzzy["fuzzy_matching"]["min_word_length"]
    EDIT_DISTANCE = config.fuzzy["fuzzy_matching"]["edit_distance"]
    MIN_SCORE = config.fuzzy["fuzzy_matching"]["min_score"]
    MIN_SIMILARITY = config.fuzzy["fuzzy_matching"]["min_similarity"]

    # INDICI NEO4J (potrebbero anche venire da config/database.yaml)
    indici = config.database.get(
        "fulltext_indexes",
        [
            {"name": "clienti_fuzzy", "label": "Cliente", "property": "name"},
            {
                "name": "fornitori_fuzzy",
                "label": "GruppoFornitore",
                "property": "ragioneSociale",
            },
            {"name": "articoli_fuzzy", "label": "Articolo", "property": "descrizione"},
            {"name": "doctype_fuzzy", "label": "DocType", "property": "name"},
            {"name": "luoghi_fuzzy", "label": "Luogo", "property": "localita"},
        ],
    )

    # ESTRAI PAROLE CANDIDATE
    parole = [p for p in question.split() if len(p) >= MIN_WORD_LENGTH]  # ← DA CONFIG

    logger.debug(f"🔍 Fuzzy matching su {len(parole)} parole: {parole}")

    # CERCA MATCH PER OGNI PAROLA
    for parola in parole:
        # SKIP parole comuni
        if parola.lower() in BLACKLIST:
            continue

        best_match = None
        best_score = 0
        best_tipo = None

        # Cerca in tutti gli indici
        for idx in indici:
            # Supporta sia formato dict che tuple per retrocompatibilità
            if isinstance(idx, dict):
                idx_name = idx["name"]
                label = idx["label"]
                prop = idx["property"]
            else:
                idx_name, label, prop = idx

            try:
                # Query con edit distance DA CONFIG
                query = f"""
                CALL db.index.fulltext.queryNodes('{idx_name}', '{parola}~{EDIT_DISTANCE}')
                YIELD node, score
                RETURN node.{prop} as nome, score
                ORDER BY score DESC
                LIMIT 1
                """
                result = graph.query(query)

                if result and result[0]["score"] > best_score:
                    best_match = result[0]["nome"]
                    best_score = result[0]["score"]
                    best_tipo = label
            except Exception as e:
                logger.debug(f"Errore ricerca fuzzy su {idx_name}: {e}")
                continue

        # APPLICA CORREZIONE SE SUPERA SOGLIE
        if best_match and best_score > MIN_SCORE:  # ← DA CONFIG
            # Check similarità caratteri
            similarity = len(set(parola.lower()) & set(best_match.lower())) / max(
                len(parola), len(best_match)
            )

            if (
                similarity > MIN_SIMILARITY and parola.lower() != best_match.lower()
            ):  # ← DA CONFIG
                question = question.replace(parola, best_match)
                correzioni.append(
                    f"'{parola}' → '{best_match}' ({best_tipo}, score={best_score:.2f}, sim={similarity:.2f})"
                )
                logger.info(f"✏️ Fuzzy: {parola} → {best_match}")

    return question, correzioni


# OTTIMIZZAZIONE 1: Caricamento schema grafo in memoria all'avvio
print("Caricamento dello schema del grafo in memoria...")
try:
    graph.refresh_schema()
    print("Schema caricato.")
    logger.info(f"Schema caricato: {graph.schema}")
except Exception as e:
    print(
        f" ERRORE: Impossibile caricare lo schema. Assicurati che Neo4j sia attivo e APOC installato. Dettagli: {e}"
    )
    logger.error(f"Errore caricamento schema: {e}")

# --- 2. TEMPLATE E FUNZIONI AUSILIARIE ---


def format_number_italian(number):
    """Formatta un numero in stile italiano: 1.234,56"""
    if isinstance(number, (int, float)):
        return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(number)


# SEZIONE 2: PROMPT DEGLI AGENTI


def load_prompt_from_yaml(file_path: str):
    """Carica il contenuto di un file YAML dato il suo percorso."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            # Ritorna l'intero dizionario se ha più chiavi,
            # o solo il valore di 'content' se presente.
            if isinstance(data, dict) and "content" in data and len(data) == 1:
                return data["content"]
            return data
    except FileNotFoundError:
        print(f"ERRORE: File prompt non trovato: {file_path}")
        return None
    except Exception as e:
        print(f"ERRORE durante il caricamento di {file_path}: {e}")
        return None


# CARICAMENTO MODULARE DEI PROMPT

PROMPTS_DIR = config.system["paths"]["prompts_dir"]

BASE_CYPHER_RULES = load_prompt_from_yaml(
    os.path.join(PROMPTS_DIR, "base_cypher_rules.yaml")
)
SPECIALIST_ROUTER_PROMPT = load_prompt_from_yaml(
    os.path.join(PROMPTS_DIR, "specialist_router.yaml")
)
SPECIALIST_CODERS = load_prompt_from_yaml(
    os.path.join(PROMPTS_DIR, "specialist_coders.yaml")
)
ADVANCED_CONTEXTUALIZER_PROMPT = load_prompt_from_yaml(
    os.path.join(PROMPTS_DIR, "contextualizer_prompt.yaml")
)
SYNTHESIZER_PROMPT = load_prompt_from_yaml(
    os.path.join(PROMPTS_DIR, "synthesizer_prompt.yaml")
)
TRANSLATOR_PROMPT = load_prompt_from_yaml(
    os.path.join(PROMPTS_DIR, "translator_prompt.yaml")
)

# Verifica che tutti i prompt siano stati caricati correttamente
if not all(
    [
        BASE_CYPHER_RULES,
        SPECIALIST_ROUTER_PROMPT,
        SPECIALIST_CODERS,
        ADVANCED_CONTEXTUALIZER_PROMPT,
        SYNTHESIZER_PROMPT,
        TRANSLATOR_PROMPT,
    ]
):
    print("Uno o più file prompt non sono stati caricati. Il programma terminerà.")
    exit()


# --- 3. FUNZIONI DEGLI AGENTI SPECIALISTI ---
def run_translator_agent(question: str) -> str:
    """Translates the user's Italian question into an English task."""
    logger.info("🤖 Chiamata all'Agente Traduttore...")
    prompt = PromptTemplate.from_template(TRANSLATOR_PROMPT)
    chain = prompt | MODELLI_STRATIFICATI["translator"] | StrOutputParser()
    task = chain.invoke({"question": question})
    return task.strip()


def run_specialist_router_agent(question: str) -> str:
    """Classifies the question to route it to the correct specialist coder."""
    logger.info("🤖 Chiamata allo Specialist Router...")
    prompt = PromptTemplate.from_template(SPECIALIST_ROUTER_PROMPT)
    chain = prompt | MODELLI_STRATIFICATI["router"] | StrOutputParser()
    route = chain.invoke({"question": question})
    return route.strip()


def run_coder_agent(question: str, relevant_schema: str, prompt_template: str) -> str:
    """
    Esegue un Coder specializzato assemblando il prompt completo prima di passarlo a LangChain.
    """
    logger.info(f"🤖 Chiamata a un Coder Specializzato...")
    # 1. PREPARO IL PROMPT HEADER CON DOMANDA E SCHEMA
    prompt_header = """
    **User Question:** {question}
    **Relevant Schema:**
    {schema}
    ---
    """
    # 2. ASSEMBLO IL PROMPT COMPLETO:
    #    Header + Regole Comuni + Regole Specifiche
    final_prompt_text = (
        prompt_header
        + BASE_CYPHER_RULES
        + "\n---\n"
        + prompt_template  # Questo è il pezzo che arriva da SPECIALIST_CODERS
        + "\nFinal Cypher Query:"
    )

    # 3. CREO IL TEMPLATE FINALE
    #   E LO PASSO A LANGCHAIN
    prompt = PromptTemplate.from_template(final_prompt_text)

    chain = prompt | MODELLI_STRATIFICATI["coder"] | StrOutputParser()

    query = chain.invoke(
        {
            "question": question,
            "schema": relevant_schema,
        }
    )
    return extract_cypher(query)


def run_contextualizer_agent(question: str, chat_history: str) -> str:
    """
    Riscrive una domanda di follow-up in una domanda autonoma, gestendo la memoria.
    """
    if not chat_history:
        return question  # Se non c'è cronologia, restituisce la domanda originale

    logger.info("Chiamata al Contextualizer Avanzato...")
    prompt = PromptTemplate.from_template(ADVANCED_CONTEXTUALIZER_PROMPT)
    chain = prompt | MODELLI_STRATIFICATI["contextualizer"] | StrOutputParser()

    rewritten_question = chain.invoke(
        {"chat_history": chat_history, "question": question}
    )
    return rewritten_question.strip()


def run_synthesizer_agent(question: str, context_str: str, total_results: int) -> str:
    logger.info("Chiamata al Synthesizer...")
    prompt = PromptTemplate.from_template(SYNTHESIZER_PROMPT)
    chain = prompt | MODELLI_STRATIFICATI["synthesizer"] | StrOutputParser()

    answer = chain.invoke(
        {
            "question": question,
            "context": context_str,
            "total_results": total_results,
        }
    )
    return answer.strip()


# FUNZIONI AUSILIARIE VARIE
def make_context_json_serializable(context):
    """
    Scorre ricorsivamente i risultati e converte TUTTI i tipi non-JSON in stringhe.
    """
    if isinstance(context, list):
        return [make_context_json_serializable(item) for item in context]
    if isinstance(context, dict):
        return {
            key: make_context_json_serializable(value) for key, value in context.items()
        }
    # Gestisce tutti i tipi di dato temporale di Neo4j e Python
    if isinstance(context, (date, Date, DateTime, Time, Duration)):
        return str(context)  # La conversione a stringa è universale e sicura
    return context


def extract_final_line(text: str) -> str:
    """
    Estrae solo l'ultima riga di testo da una potenziale risposta multi-riga di un LLM.
    È una misura di sicurezza contro la prolissità.
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def extract_cypher(text: str) -> str:
    """
    Estrae la query Cypher da una stringa, anche se è avvolta in blocchi di codice Markdown.
    """
    # Cerca un blocco di codice Cypher
    match = re.search(r"```(?:cypher)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Se non trova un blocco, assume che la risposta sia già la query pulita
    return text.strip()


# --- 4. ENDPOINT API CON CACHE ---


class UserQuestionWithSession(BaseModel):
    question: str
    user_id: Optional[str] = (
        "default_user"  # Identificativo utente opzionale per la sessione
    )


# =======================================================================
@app.get("/health")
async def health_check():
    """Endpoint per verificare lo stato dell'API"""
    try:
        # Ottieni lo stato del database Neo4j
        graph.query("RETURN 1")
        return {
            "status": "online",
            "message": "API BlueAI operativa",
            "neo4j": "connected",
        }
    except Exception as e:
        logger.error(f"Health check fallito: {e}")
        return {"status": "error", "message": str(e), "neo4j": "disconnected"}


# ENDPOINT PER LE DOMANDE
@app.post("/ask")
async def ask_question(user_question: UserQuestionWithSession):
    start_total_time = time.perf_counter()
    timing_details = {}

    user_id = user_question.user_id or "default_user"
    original_question_text = user_question.question.strip()
    session = get_or_create_session(user_id)

    logger.info(f"📥 [{user_id}] Domanda: '{original_question_text}'")

    try:
        ## == FASE 1: PRE-PROCESSING E GESTIONE MEMORIA
        start_preprocessing_time = time.perf_counter()

        # 1a. Gestione Memoria: Il nuovo Contextualizer riscrive la domanda
        chat_history = get_conversation_context(user_id)
        question_text = run_contextualizer_agent(original_question_text, chat_history)
        if question_text != original_question_text:
            logger.info(f"Domanda riscritta dal Contestualizzatore: '{question_text}'")

        # 1b. Correzione Fuzzy
        question_text, correzioni = correggi_nomi_fuzzy_neo4j(question_text)
        if correzioni:
            logger.info(f"Correzioni Fuzzy applicate: {', '.join(correzioni)}")

        # 1c. Traduzione (fondamentale per la coerenza dei prompt)
        english_task = run_translator_agent(question_text)
        logger.info(f"Task tradotto in Inglese: '{english_task}'")

        timing_details["preprocessing"] = time.perf_counter() - start_preprocessing_time

        ## == FASE 2: ROUTING E GENERAZIONE QUERY SPECIALIZZATA

        # 2a. Il Router decide quale specialista usare
        start_route_time = time.perf_counter()
        specialist_route = run_specialist_router_agent(english_task)
        timing_details["routing"] = time.perf_counter() - start_route_time
        logger.info(f" Rotta decisa dallo Specialist Router: {specialist_route}")

        # 2b. Seleziona il prompt corretto dal dizionario degli specialisti
        coder_prompt_template = SPECIALIST_CODERS.get(
            specialist_route, SPECIALIST_CODERS["GENERAL_QUERY"]
        )

        relevant_schema = graph.schema

        # 2c. Genera la query Cypher con il Coder specializzato
        start_gen_time = time.perf_counter()
        generated_query = run_coder_agent(
            question=english_task,
            relevant_schema=relevant_schema,
            prompt_template=coder_prompt_template,
        )
        timing_details["generazione_query"] = time.perf_counter() - start_gen_time
        logger.info(
            f"Query generata dallo specialista '{specialist_route}':\n{generated_query}"
        )

        ## == FASE 3: ESECUZIONE E SINTESI DELLA RISPOSTA
        # 3a. Esecuzione della query sul database
        start_db_time = time.perf_counter()
        context_completo = graph.query(generated_query)
        total_results = len(context_completo)
        timing_details["esecuzione_db"] = time.perf_counter() - start_db_time
        logger.info(f"Query eseguita. Trovati {total_results} risultati.")

        # 3b. Prepara il contesto e chiama il Synthesizer per la risposta finale
        MAX_RESULTS_FOR_SYNTHESIZER = 10  # Aumentato a 10 per risposte più ricche
        context_da_inviare = context_completo[:MAX_RESULTS_FOR_SYNTHESIZER]

        sanitized_context = make_context_json_serializable(context_da_inviare)
        context_str_for_llm = json.dumps(
            {"risultato": sanitized_context}, indent=2, ensure_ascii=False
        )

        start_synth_time = time.perf_counter()
        final_answer = run_synthesizer_agent(
            question=original_question_text,
            context_str=context_str_for_llm,
            total_results=total_results,
        )
        timing_details["sintesi_risposta"] = time.perf_counter() - start_synth_time

        ## == FASE 4: FINALIZZAZIONE E SALVATAGGIO DELLA RISPOSTA
        timing_details["totale"] = time.perf_counter() - start_total_time
        logger.info(
            f"Risposta Finale generata in {timing_details['totale']:.2f}s: {final_answer}"
        )

        add_message_to_session(
            user_id,
            original_question_text,
            final_answer,
            context_completo,
            generated_query,
        )

        return {
            "domanda": original_question_text,
            "query_generata": generated_query,
            "risposta": final_answer,
            "timing_details": timing_details,
        }

    except Exception as e:
        logger.error(
            f"ERRORE GRAVE nel flusso ad agenti specializzati: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Errore durante l'elaborazione: {e}"
        )


# Endpoint per ottenere la cronologia della conversazione
@app.get("/conversation/{user_id}")
async def get_conversation(user_id: str):
    """Ottieni la cronologia della conversazione per un utente"""
    session = conversation_sessions.get(user_id)
    if not session:
        return {"messages": [], "total": 0}

    return {
        "user_id": user_id,
        "messages": [
            {
                "timestamp": msg.timestamp.isoformat(),
                "question": msg.question,
                "answer": msg.answer,
                "had_context": len(msg.context) > 0,
            }
            for msg in session.messages
        ],
        "total": len(session.messages),
        "session_created": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
    }


# Endpoint per cancellare la memoria di un utente
@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Cancella la cronologia conversazione di un utente"""
    if user_id in conversation_sessions:
        del conversation_sessions[user_id]
        return {"message": f"Conversazione cancellata per {user_id}"}
    return {"message": f"Nessuna conversazione trovata per {user_id}"}


@app.delete("/cache")
async def clear_cache():
    """
    Endpoint per svuotare TUTTE le cache: risposte e memoria conversazioni.
    """
    global query_cache
    global conversation_sessions

    logger.info(" Richiesta di svuotamento manuale di TUTTE le cache...")
    try:
        # Svuota la cache delle risposte
        responses_removed = query_cache.currsize
        query_cache.clear()

        # Svuota la cache delle sessioni di conversazione
        sessions_removed = len(conversation_sessions)
        conversation_sessions.clear()

        logger.info(
            f"Cache svuotata con successo. (Risposte rimosse: {responses_removed}, Sessioni rimosse: {sessions_removed})"
        )
        return {
            "message": "Tutte le cache (risposte e conversazioni) sono state svuotate.",
            "risposte_rimosse": responses_removed,
            "sessioni_rimosse": sessions_removed,
        }
    except Exception as e:
        logger.error(f" Errore durante lo svuotamento della cache: {e}")
        return {
            "message": "Errore durante lo svuotamento della cache.",
            "error": str(e),
        }


# =======================================================================
