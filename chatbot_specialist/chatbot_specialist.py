from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_neo4j import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
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
from neo4j.exceptions import Neo4jError
from pathlib import Path


# Importazione dei modelli LLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from google.cloud import aiplatform

# Carica .env
load_dotenv()


from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from google.cloud import aiplatform
from pydantic import PrivateAttr


class VertexAIDedicatedEndpoint(BaseChatModel):
    """Wrapper per endpoint dedicati Vertex AI"""

    project_id: str
    location: str
    endpoint_id: str
    temperature: float = 0.0
    max_output_tokens: int = 2048

    # Attributo privato per l'endpoint (non parte del modello Pydantic)
    _endpoint: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Inizializza l'endpoint dopo che Pydantic ha validato i campi"""
        super().model_post_init(__context)
        aiplatform.init(project=self.project_id, location=self.location)
        self._endpoint = aiplatform.Endpoint(self.endpoint_id)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Genera una risposta"""
        # Converti i messaggi in formato Vertex AI
        prompt = self._messages_to_prompt(messages)

        # Chiama l'endpoint dedicato
        response = self._endpoint.predict(
            instances=[{"prompt": prompt}],
            parameters={
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
            use_dedicated_endpoint=True,  # CRITICO per endpoint dedicati!
        )

        # Estrai il testo dalla risposta
        text = response.predictions[0] if response.predictions else ""

        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Converti i messaggi langchain in un prompt"""
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")
        return "\n\n".join(prompt_parts)

    @property
    def _llm_type(self) -> str:
        return "vertex-ai-dedicated"


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
        self.prompt_profiles = self._load_yaml("prompt_profiles.yaml") or {}

    def get_system_message(self, agent_name: str) -> str:
        """Ritorna il system message per provider+agente dalle config.

        Ordine:
          profiles[provider][agent] → defaults[provider] → generic fallback
        """
        try:
            cfg = self.get_model_config(agent_name)
            provider = cfg["provider"]
            prof = (self.prompt_profiles or {}).get("profiles", {})
            defaults = (self.prompt_profiles or {}).get("defaults", {})
            if provider in prof and agent_name in prof[provider]:
                return prof[provider][agent_name] or defaults.get(provider) or ""
            return defaults.get(provider) or ""
        except Exception:
            return ""

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
            **model_def,
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

    elif cfg["provider"] == "vertex_ai":
        return VertexAIDedicatedEndpoint(
            project_id=cfg["project_id"],
            location=cfg["location"],
            endpoint_id=cfg["endpoint_id"],
            temperature=cfg["provider_config"]["temperature"],
        )

    elif cfg["provider"] == "google":
        api_key = config.get_secret(cfg["provider_config"]["api_key_env"])
        return ChatGoogleGenerativeAI(
            model=cfg["model_name"],
            google_api_key=api_key,
            temperature=cfg["provider_config"]["temperature"],
            convert_system_message_to_human=False,
        )

    else:
        raise ValueError(f"Provider sconosciuto: {cfg['provider']}")


# Creazione modelli stratificati
MODELLI_STRATIFICATI = {
    agent: create_llm_model(agent)
    for agent in [
        "contextualizer",
        "router",
        "coder",
        "synthesizer",
        "translator",
        "general_conversation",
    ]
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


# == Query Repair diagnostics helpers ==
def _diagnostics_dir() -> Path:
    base = config.system.get("logging", {}).get("diagnostics_dir", "diagnostics")
    p = Path(base)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _load_recent_repairs(max_items: int) -> str:
    """Carica ultimi fix da file JSONL per includerli nel prompt del repair."""
    diag_dir = _diagnostics_dir()
    path = diag_dir / "query_repair_log.jsonl"
    if not path.exists():
        return ""
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines()[-max_items:]:
                try:
                    rec = json.loads(line)
                    # Sintesi compatta
                    lines.append(
                        f"- err: {rec.get('error_short','')} | fix: {rec.get('fixed_hint','')}"
                    )
                except Exception:
                    continue
    except Exception:
        return ""
    return "\n".join(lines)


def _append_repair_event(question: str, bad_query: str, error: str, fixed_query: str):
    if not config.system.get("logging", {}).get("save_diagnostics", False):
        return
    diag_dir = _diagnostics_dir()
    path = diag_dir / "query_repair_log.jsonl"
    rec = {
        "ts": datetime.now().isoformat(),
        "question": question,
        "bad_query": bad_query,
        "error": error,
        "error_short": (error.split("\n", 1)[0] if error else ""),
        "fixed_query": fixed_query,
        # Heuristic short hint
        "fixed_hint": (fixed_query.split("\n", 1)[0] if fixed_query else ""),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _suggest_unnecessary_with_hints(cypher: str) -> str:
    """Detects potentially unnecessary WITH clauses and returns textual hints.

    Heuristic: flags lines starting with WITH that
    - do not contain DISTINCT/ORDER BY/LIMIT/WHERE
    - do not contain function calls ("(")
    These often act as pass-through and can be removed or merged.
    """
    try:
        hints = []
        # 0) Flag SQL GROUP BY usage
        if re.search(r"\bGROUP\s+BY\b", cypher, flags=re.IGNORECASE):
            hints.append(
                "SQL keyword 'GROUP BY' detected: Cypher does not support GROUP BY. Replace with a WITH clause listing grouping keys, then ORDER/RETURN using aggregated values."
            )
        lines = cypher.splitlines()
        for idx, line in enumerate(lines, start=1):
            m = re.match(r"^\s*WITH\s+(.*)$", line, flags=re.IGNORECASE)
            if not m:
                continue
            rest = (m.group(1) or "").strip().rstrip(";")
            up = rest.upper()
            if any(tok in up for tok in ["DISTINCT", "ORDER BY", "LIMIT", "WHERE"]):
                continue
            if "(" in rest:
                # likely aggregations or function calls
                continue
            # Simple pass-through WITH — suggest removal or merge
            hints.append(
                f"Line {idx}: '{line.strip()}' looks unnecessary; consider removing or merging into a single MATCH ... RETURN."
            )
        if hints:
            return "Auto-detected issues and suggestions:\n- " + "\n- ".join(hints)
    except Exception:
        pass
    return ""


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


class QueryTimeoutError(RuntimeError):
    """Raised when a Neo4j query exceeds the configured timeout."""


def _parse_positive_float(value: Optional[float]) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if parsed <= 0:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def get_query_timeout_seconds() -> Optional[float]:
    """Return the configured Neo4j query timeout in seconds, if any."""

    try:
        ql_cfg = config.system.get("query_limits", {}) or {}
        return _parse_positive_float(ql_cfg.get("query_timeout_seconds"))
    except Exception:
        return None


def execute_cypher_with_timeout(
    query: str, params: Optional[Dict[str, Any]] = None, timeout_seconds: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Execute a Cypher query enforcing the configured timeout."""

    effective_timeout = timeout_seconds
    if effective_timeout is None:
        effective_timeout = get_query_timeout_seconds()

    timeout_ms = None
    if effective_timeout:
        timeout_ms = max(1, int(effective_timeout * 1000))

    run_kwargs = {}
    if timeout_ms:
        run_kwargs["timeout"] = timeout_ms

    try:
        with graph._driver.session() as session:  # type: ignore[attr-defined]
            result = session.run(query, params or {}, **run_kwargs)
            return [record.data() for record in result]
    except Neo4jError as exc:
        code = getattr(exc, "code", "") or ""
        message = str(exc)
        if "Timeout" in code or "timed out" in message.lower():
            if effective_timeout:
                human_msg = (
                    f"Neo4j query exceeded the {effective_timeout:.0f}s timeout limit."
                )
            else:
                human_msg = "Neo4j query timed out during execution."
            raise QueryTimeoutError(human_msg) from exc
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
                result = execute_cypher_with_timeout(query)

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


def resolve_prompt_path(agent_name: str, filename: str) -> str:
    """Seleziona il file prompt in base a modello/provider, con fallback al default.

    Ordine tentativi:
      1) prompts/<model_name>/<filename>
      2) prompts/<provider>/<filename>
      3) prompts/<filename>
    """
    prompts_dir = config.system["paths"]["prompts_dir"]
    cfg = config.get_model_config(agent_name)
    model_name = cfg.get("model_name")
    model_id = cfg.get("model_id")
    provider = cfg.get("provider")

    # Support model aliasing from config/prompt_profiles.yaml
    aliases = (config.prompt_profiles or {}).get("model_aliases", {})
    alias_dir = aliases.get(model_name) or aliases.get(model_id)

    candidates = []
    # 1) exact model_name directory
    if model_name:
        candidates.append(os.path.join(prompts_dir, model_name, filename))
    # 1b) alias directory (e.g., map llama3-8b-vertex → llama3)
    if alias_dir:
        candidates.append(os.path.join(prompts_dir, alias_dir, filename))
    # 2) provider directory
    if provider:
        candidates.append(os.path.join(prompts_dir, provider, filename))
    # 3) default
    candidates.append(os.path.join(prompts_dir, filename))
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback finale: ultimo della lista
    return candidates[-1]


# CARICAMENTO MODULARE DEI PROMPT (con varianti per modello/provider)
PROMPTS_DIR = config.system["paths"]["prompts_dir"]

BASE_CYPHER_RULES = load_prompt_from_yaml(
    resolve_prompt_path("coder", "base_cypher_rules.yaml")
)
SPECIALIST_ROUTER_PROMPT = load_prompt_from_yaml(
    resolve_prompt_path("router", "specialist_router.yaml")
)
SPECIALIST_CODERS = load_prompt_from_yaml(
    resolve_prompt_path("coder", "specialist_coders.yaml")
)
ADVANCED_CONTEXTUALIZER_PROMPT = load_prompt_from_yaml(
    resolve_prompt_path("contextualizer", "contextualizer_prompt.yaml")
)
SYNTHESIZER_PROMPT = load_prompt_from_yaml(
    resolve_prompt_path("synthesizer", "synthesizer_prompt.yaml")
)
TRANSLATOR_PROMPT = load_prompt_from_yaml(
    resolve_prompt_path("translator", "translator_prompt.yaml")
)
GENERAL_CONVERSATION_PROMPT = load_prompt_from_yaml(
    resolve_prompt_path("general_conversation", "general_conversation_prompt.yaml")
)
QUERY_REPAIR_PROMPT = load_prompt_from_yaml(
    resolve_prompt_path("coder", "query_repair_prompt.yaml")
)


def _invoke_with_profile(agent_name: str, prompt_text: str, variables: dict) -> str:
    """Invoca l'LLM con una struttura di prompt diversa per provider."""
    cfg = config.get_model_config(agent_name)
    model = MODELLI_STRATIFICATI[agent_name]

    try:
        if cfg["provider"] in ("openai", "ollama", "google"):
            system_msg = config.get_system_message(agent_name)
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_msg), ("user", prompt_text)]
            )
            chain = prompt | model | StrOutputParser()
            return chain.invoke(variables)
        else:
            # Fallback per altri provider (es. Vertex AI)
            prompt = PromptTemplate.from_template(prompt_text)
            chain = prompt | model | StrOutputParser()
            return chain.invoke(variables)

    except Exception as e:
        # Log dell'errore e rilancia (oppure gestisci in modo più specifico)
        logger.error(f"Errore invocando {agent_name}: {e}")
        raise


# Verifica che tutti i prompt siano stati caricati correttamente
if not all(
    [
        BASE_CYPHER_RULES,
        SPECIALIST_ROUTER_PROMPT,
        SPECIALIST_CODERS,
        ADVANCED_CONTEXTUALIZER_PROMPT,
        SYNTHESIZER_PROMPT,
        TRANSLATOR_PROMPT,
        GENERAL_CONVERSATION_PROMPT,
    ]
):
    print("Uno o più file prompt non sono stati caricati. Il programma terminerà.")
    exit()

logger.info("📚 Inizializzo Example Retriever...")
from example_retriever import ExampleRetriever

example_retriever = ExampleRetriever()
logger.info(f"✅ Esempi: {example_retriever.get_stats()}")


# --- 3. FUNZIONI DEGLI AGENTI SPECIALISTI ---
def run_translator_agent(question: str) -> str:
    """Translates the user's Italian question into an English task."""
    logger.info("🤖 Chiamata all'Agente Traduttore...")
    # Usa adattatore per differenziare struttura tra modelli
    task = _invoke_with_profile(
        agent_name="translator",
        prompt_text=TRANSLATOR_PROMPT,
        variables={"question": question},
    )
    return task.strip()


def run_general_conversation_agent(question: str) -> str:
    """Generates a conversational answer without querying the graph."""
    logger.info("🤖 Chiamata all'Agente Conversazionale...")
    response = _invoke_with_profile(
        agent_name="general_conversation",
        prompt_text=GENERAL_CONVERSATION_PROMPT,
        variables={"question": question},
    )
    return response.strip()


def run_specialist_router_agent(question: str) -> str:
    """Classifies the question to route it to the correct specialist coder."""
    logger.info("🤖 Chiamata allo Specialist Router...")
    route = _invoke_with_profile(
        agent_name="router",
        prompt_text=SPECIALIST_ROUTER_PROMPT,
        variables={"question": question},
    )
    return route.strip()


def run_coder_agent_2(question: str, relevant_schema: str, prompt_template: str) -> str:
    """Esegue un Coder specializzato assemblando il prompt completo prima di passarlo a LangChain."""
    logger.info(f"🤖 Chiamata a un Coder Specializzato...")
    # 1. Header/Suffix da config
    coder_tpl = (config.prompt_profiles or {}).get("templates", {}).get("coder", {})
    prompt_header = coder_tpl.get("header", "{schema}\n---\n")
    prompt_suffix = coder_tpl.get("suffix", "Final Cypher Query:")

    # 2. Assemblaggio: Header + Regole comuni + Regole speciali + Suffix
    final_prompt_text = (
        prompt_header
        + BASE_CYPHER_RULES
        + "\n---\n"
        + prompt_template
        + "\n"
        + prompt_suffix
    )

    # 3. CREO IL TEMPLATE FINALE
    #   E LO PASSO A LANGCHAIN
    query = _invoke_with_profile(
        agent_name="coder",
        prompt_text=final_prompt_text,
        variables={"question": question, "schema": relevant_schema},
    )
    candidate = extract_cypher(query)
    # Pre-repair guardrails: if SQL artifacts or non-Cypher preface slipped in, trigger a repair pass
    needs_repair = bool(re.search(r"\bGROUP\s+BY\b", candidate, flags=re.IGNORECASE))
    if needs_repair and QUERY_REPAIR_PROMPT:
        try:
            fixed = _invoke_with_profile(
                agent_name="coder",
                prompt_text=QUERY_REPAIR_PROMPT,
                variables={
                    "question": question,
                    "schema": relevant_schema,
                    "bad_query": candidate,
                    "error": "Invalid SQL dialect: found GROUP BY; rewrite using Cypher WITH.",
                    "hints": _suggest_unnecessary_with_hints(candidate),
                    "recent_repairs": "",
                },
            )
            candidate = extract_cypher(fixed)
        except Exception:
            pass
    return candidate


def run_coder_agent(
    question: str,
    relevant_schema: str,
    prompt_template: str,
    specialist_type: str,
    original_question_it: str,
) -> str:
    """Coder con esempi dinamici RAG"""
    logger.info(f"🤖 Coder: {specialist_type}")

    # 1. RECUPERA ESEMPI
    # Usa la domanda originale in IT per il retrieval degli esempi
    relevant_examples = example_retriever.retrieve(
        question=original_question_it, specialist=specialist_type, top_k=3
    )

    # 2. FORMATTA ESEMPI (con escape delle graffe)
    if relevant_examples:
        examples_text = "\n\n**RELEVANT EXAMPLES:**\n" + "\n\n".join(
            [
                f"Q: {ex['question']}\n```cypher\n{ex['cypher'].strip().replace('{', '{{').replace('}', '}}')}\n```"
                for ex in relevant_examples
            ]
        )
        logger.info(f"📚 Recuperati: {[ex['id'] for ex in relevant_examples]}")
    else:
        examples_text = ""

    # 3. ASSEMBLA PROMPT
    prompt_header = """
    **Original Question (IT):** {original_question_it}
    **Task (EN):** {question}
    **Relevant Schema:**
    {schema}
    ---
    """

    final_prompt_text = (
        prompt_header
        + BASE_CYPHER_RULES
        + "\n---\n"
        + prompt_template
        + "\n---\n"
        + examples_text
        + "\nFinal Cypher Query:"
    )

    # 4. GENERA (passa attraverso il profilo per includere il system message)
    query = _invoke_with_profile(
        agent_name="coder",
        prompt_text=final_prompt_text,
        variables={
            "question": question,  # EN
            "schema": relevant_schema,
            "original_question_it": original_question_it,
        },
    )

    return extract_cypher(query)


def run_contextualizer_agent(question: str, chat_history: str) -> str:
    """
    Riscrive una domanda di follow-up in una domanda autonoma, gestendo la memoria.
    """
    if not chat_history:
        return question  # Se non c'è cronologia, restituisce la domanda originale

    logger.info("Chiamata al Contextualizer Avanzato...")
    rewritten_question = _invoke_with_profile(
        agent_name="contextualizer",
        prompt_text=ADVANCED_CONTEXTUALIZER_PROMPT,
        variables={"chat_history": chat_history, "question": question},
    )
    return rewritten_question.strip()


def run_synthesizer_agent(question: str, context_str: str, total_results: int) -> str:
    logger.info("Chiamata al Synthesizer...")
    answer = _invoke_with_profile(
        agent_name="synthesizer",
        prompt_text=SYNTHESIZER_PROMPT,
        variables={
            "question": question,
            "context": context_str,
            "total_results": total_results,
        },
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
    if not text:
        return ""
    t = text.strip()
    # 1) Cerca un blocco di codice Cypher
    match = re.search(r"```(?:cypher)?\s*\n(.*?)\n\s*```", t, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 2) Rimuovi eventuali prefazioni (es. "Here is the corrected Cypher query:")
    lines = t.splitlines()
    cypher_start = re.compile(
        r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|CALL|UNWIND|MERGE|CREATE)\b",
        re.IGNORECASE,
    )
    start_idx = None
    for i, line in enumerate(lines):
        if cypher_start.search(line):
            start_idx = i
            break
    if start_idx is not None:
        candidate = "\n".join(lines[start_idx:]).strip()
        candidate = candidate.strip("`")
        return candidate
    # 3) Fallback: restituisce il testo così com'è
    return t


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
        execute_cypher_with_timeout("RETURN 1")
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

        if specialist_route == "GENERAL_CONVERSATION":
            start_conv_time = time.perf_counter()
            conversation_reply = run_general_conversation_agent(original_question_text)
            timing_details["conversazione"] = time.perf_counter() - start_conv_time
            timing_details["generazione_query"] = 0.0
            timing_details["esecuzione_db"] = 0.0
            timing_details["sintesi_risposta"] = 0.0
            timing_details["totale"] = time.perf_counter() - start_total_time

            add_message_to_session(
                user_id,
                original_question_text,
                conversation_reply,
                [],
                "",
            )

            return {
                "domanda": original_question_text,
                "query_generata": None,
                "risposta": conversation_reply,
                "timing_details": timing_details,
            }

        # 2b. Seleziona il prompt corretto dal dizionario degli specialisti
        coder_prompt_template = SPECIALIST_CODERS.get(
            specialist_route, SPECIALIST_CODERS["GENERAL_QUERY"]
        )

        relevant_schema = graph.schema

        # 2c. Genera la query Cypher con il Coder specializzato
        start_gen_time = time.perf_counter()
        """
        generated_query = run_coder_agent(
            question=english_task,
            relevant_schema=relevant_schema,
            prompt_template=coder_prompt_template,
        )
        """
        generated_query = run_coder_agent(
            question=english_task,
            relevant_schema=relevant_schema,
            prompt_template=coder_prompt_template,
            specialist_type=specialist_route,
            original_question_it=original_question_text,
        )

        timing_details["generazione_query"] = time.perf_counter() - start_gen_time
        logger.info(
            f"Query generata dallo specialista '{specialist_route}':\n{generated_query}"
        )

        ## == FASE 3: ESECUZIONE E SINTESI DELLA RISPOSTA
        # 3a. Esecuzione query con ciclo di repair (configurabile)
        start_db_time = time.perf_counter()
        attempts = max(1, int(config.system.get("retry", {}).get("max_attempts", 1)))
        patterns = config.system.get("retry", {}).get("repairable_error_patterns", [])
        max_recent = int(config.system.get("retry", {}).get("max_recent_repairs", 0))
        hints_texts = []
        for hint_file in config.system.get("retry", {}).get("hints_files", []) or []:
            try:
                with open(hint_file, "r", encoding="utf-8") as hf:
                    hints_texts.append(hf.read())
            except Exception:
                continue
        recent_repairs = _load_recent_repairs(max_recent) if max_recent > 0 else ""

        last_error_msg = None
        current_query = generated_query

        # Preflight: intercetta sintassi SQL non valida per Cypher (es. GROUP BY, HAVING, OVER)
        try:
            sql_like = re.compile(
                r"\b(GROUP\s+BY|HAVING|OVER\b|PARTITION\s+BY|ROW_NUMBER\s*\(|RANK\s*\(|WINDOW\b)",
                re.IGNORECASE,
            )
            if current_query and sql_like.search(current_query) and QUERY_REPAIR_PROMPT:
                logger.info(
                    "Preflight Repair: rilevata sintassi SQL-like (GROUP BY/HAVING/OVER). Provo a correggere prima dell'esecuzione."
                )
                preflight_hints = (
                    "SQL-like keywords are invalid in Cypher (GROUP BY, HAVING, OVER).\n"
                    "Use Cypher aggregation instead: RETURN key, sum(val) AS total; or WITH key, sum(val) AS total RETURN key, total.\n"
                    "For best-year or ranking, use WITH + collect(...) and pick the first element after ORDER BY, not GROUP BY.\n"
                )
                auto_hints_pf = _suggest_unnecessary_with_hints(current_query)
                combined_hints_pf = "\n\n".join(
                    [
                        h
                        for h in [
                            hints_blob if "hints_blob" in locals() else "",
                            preflight_hints,
                            auto_hints_pf,
                        ]
                        if h
                    ]
                ).strip()
                fixed_pf = _invoke_with_profile(
                    agent_name="coder",
                    prompt_text=QUERY_REPAIR_PROMPT,
                    variables={
                        "question": english_task,
                        "schema": relevant_schema,
                        "bad_query": current_query,
                        "error": "Detected SQL-like syntax (GROUP BY/HAVING/OVER) not valid in Cypher.",
                        "hints": combined_hints_pf,
                        "recent_repairs": recent_repairs,
                    },
                )
                fixed_pf_query = extract_cypher(fixed_pf)
                logger.info(f"Query corretta (preflight) proposta:\n{fixed_pf_query}")
                _append_repair_event(
                    english_task,
                    current_query,
                    "Preflight SQL-like syntax",
                    fixed_pf_query,
                )
                current_query = fixed_pf_query
        except Exception:
            pass
        for attempt_idx in range(1, attempts + 1):
            try:
                context_completo = execute_cypher_with_timeout(current_query)
                generated_query = current_query
                last_error_msg = None
                break
            except QueryTimeoutError as timeout_exc:
                last_error_msg = str(timeout_exc)
                logger.error(
                    f"Tentativo {attempt_idx}/{attempts} fallito per timeout della query Neo4j: {last_error_msg}"
                )
                raise
            except Exception as e:
                last_error_msg = str(e)
                logger.warning(
                    f"Tentativo {attempt_idx}/{attempts} fallito: {last_error_msg[:180]}"
                )
                if not QUERY_REPAIR_PROMPT:
                    continue
                # Check se l'errore è riparabile
                if not any(
                    (
                        re.search(pat, last_error_msg)
                        if any(ch in pat for ch in ".*?[]()^$")
                        else (pat in last_error_msg)
                    )
                    for pat in patterns
                ):
                    continue
                # Prepara prompt di repair
                hints_blob = "\n\n".join(hints_texts).strip()
                # Auto-detected hints (e.g., unnecessary WITH)
                auto_hints = _suggest_unnecessary_with_hints(current_query)
                combined_hints = "\n\n".join(hints_texts).strip() or ""
                if auto_hints:
                    combined_hints = (
                        (combined_hints + "\n\n" + auto_hints).strip()
                        if combined_hints
                        else auto_hints
                    )

                fixed = _invoke_with_profile(
                    agent_name="coder",
                    prompt_text=QUERY_REPAIR_PROMPT,
                    variables={
                        "question": english_task,
                        "schema": relevant_schema,
                        "bad_query": current_query,
                        "error": last_error_msg,
                        "hints": combined_hints,
                        "recent_repairs": recent_repairs,
                    },
                )
                fixed_query = extract_cypher(fixed)
                logger.info(f"Query corretta proposta:\n{fixed_query}")
                _append_repair_event(
                    english_task, current_query, last_error_msg, fixed_query
                )
                current_query = fixed_query
                continue
        if last_error_msg:
            # re-raise ultimo errore se tutti i tentativi falliscono
            raise Exception(last_error_msg)
        total_results = len(context_completo)
        timing_details["esecuzione_db"] = time.perf_counter() - start_db_time
        logger.info(f"Query eseguita. Trovati {total_results} risultati.")

        # 3b. Prepara il contesto e chiama il Synthesizer per la risposta finale
        # Limite gestito da config: query_limits.provider_overrides.synthesizer
        synth_cfg = config.get_model_config("synthesizer")
        ql = config.system.get("query_limits", {})
        provider_overrides = ql.get("provider_overrides", {}).get("synthesizer", {})
        max_results_cfg = ql.get("max_results_for_synthesizer", 10)
        max_results_final = int(
            provider_overrides.get(synth_cfg["provider"], max_results_cfg)
        )
        context_da_inviare = context_completo[:max_results_final]

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

    except QueryTimeoutError as e:
        logger.error(
            f"Timeout durante l'esecuzione della query Neo4j: {e}", exc_info=False
        )
        raise HTTPException(status_code=504, detail=str(e))
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


@app.post("/debug/rag")
async def debug_rag(request: dict):
    """
    Testa il retrieval RAG per una domanda

    Body:
    {
        "question": "Chi è il cliente con maggior fatturato?",
        "specialist": "SALES_CYCLE",  # opzionale, se omesso usa router
        "top_k": 5  # opzionale, default 3
    }
    """
    question = request.get("question")
    specialist = request.get("specialist")
    top_k = request.get("top_k", 3)

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    # Se specialist non specificato, usa router
    if not specialist:
        english_task = run_translator_agent(question)
        specialist = run_specialist_router_agent(english_task)

    # Recupera esempi
    examples = example_retriever.retrieve(question, specialist, top_k=top_k)

    return {
        "question": question,
        "specialist_used": specialist,
        "top_k": top_k,
        "retrieved_examples": [
            {
                "rank": i + 1,
                "id": ex["id"],
                "question": ex["question"],
                "cypher": ex["cypher"],
            }
            for i, ex in enumerate(examples)
        ],
        "total_available": len(
            example_retriever.examples_by_specialist.get(specialist, [])
        ),
    }


# =======================================================================
