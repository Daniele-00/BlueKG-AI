from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_neo4j import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from cachetools import TTLCache
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Literal, Pattern
from pydantic import BaseModel
import re
import logging
import time
import json
from datetime import date
import copy
import hashlib
import numpy as np
from neo4j.time import Date, DateTime, Time, Duration
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node, Relationship, Path
from pathlib import Path


# Importazione dei modelli LLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from groqWrapper import ChatGroq
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

    elif cfg["provider"] == "groq":
        api_key = config.get_secret(cfg["provider_config"]["api_key_env"])
        return ChatGroq(
            model=cfg["model_name"],
            groq_api_key=api_key,
            temperature=cfg["provider_config"]["temperature"],
            max_tokens=cfg.get("max_tokens", 2000),
            timeout=cfg["provider_config"]["timeout"],
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

# Flag cache/memoria
ENABLE_CACHE = config.system.get("cache", {}).get("enabled", True)

SESSION_CFG = config.system.get("sessions", {}) or {}
ENABLE_MEMORY = bool(SESSION_CFG.get("enable_memory", False))
SESSION_TIMEOUT = timedelta(minutes=SESSION_CFG.get("timeout_minutes", 30))
MAX_MESSAGES_PER_SESSION = int(SESSION_CFG.get("max_messages_per_session", 15))

MEMORY_CFG = SESSION_CFG.get("memory", {}) or {}

EXAMPLES_CFG = config.system.get("examples", {}) or {}
EXAMPLES_MIN_SIMILARITY = EXAMPLES_CFG.get("min_similarity")

RETRY_DEFAULT_CFG = config.system.get("retry", {}) or {}
SEMANTIC_EXPANSION_CFG = RETRY_DEFAULT_CFG.get("semantic_expansion", {}) or {}
SEMANTIC_EXPANSION_ENABLED = bool(SEMANTIC_EXPANSION_CFG.get("enabled", False))
SEMANTIC_EXPANSION_MAX_ATTEMPTS_DEFAULT = int(
    SEMANTIC_EXPANSION_CFG.get("max_attempts", 0) or 0
)
SEMANTIC_EXPANSION_TOP_K_DEFAULT = int(
    SEMANTIC_EXPANSION_CFG.get("top_k_examples", 0) or 0
)
SEMANTIC_EXPANSION_PROMPT_NAME = SEMANTIC_EXPANSION_CFG.get(
    "prompt", "query_expansion_prompt.yaml"
)


def _normalize_keyword_list(values) -> List[str]:
    collection = values or []
    return [str(item).strip().lower() for item in collection if str(item).strip()]


def _contains_keyword(lowered: str, tokens: List[str], keywords: Set[str]) -> bool:
    """Verifica se la frase contiene una delle keyword (single word o multi-word)."""

    if not keywords:
        return False

    token_set = set(tokens)
    for keyword in keywords:
        if not keyword:
            continue
        if " " in keyword:
            if keyword in lowered:
                return True
        else:
            if keyword in token_set:
                return True
    return False


SMALL_TALK_CFG = config.system.get("small_talk", {}) or {}
SMALL_TALK_KEYWORDS = set(_normalize_keyword_list(SMALL_TALK_CFG.get("keywords")))
DANGEROUS_CFG = config.system.get("dangerous_operations", {}) or {}
DANGEROUS_KEYWORDS = set(_normalize_keyword_list(DANGEROUS_CFG.get("keywords")))
CYPHER_WRITE_BLOCKLIST = [
    str(item).strip()
    for item in config.system.get("cypher_write_blocklist", []) or []
    if str(item).strip()
]
QUERY_SAFETY_CFG = config.system.get("query_safety", {}) or {}
QUERY_RISKY_TIMEOUT = float(QUERY_SAFETY_CFG.get("risky_timeout_seconds", 12) or 12)
QUERY_CRITICAL_TIMEOUT = float(QUERY_SAFETY_CFG.get("critical_timeout_seconds", 8) or 8)
SAFE_REWRITE_LIMIT = int(QUERY_SAFETY_CFG.get("safe_rewrite_limit", 200) or 200)
SLOW_QUERY_LOG_PATH = QUERY_SAFETY_CFG.get(
    "slow_query_log", "diagnostics/slow_queries.log"
)

FEEDBACK_CFG = config.system.get("feedback", {}) or {}
FEEDBACK_ENABLED = bool(FEEDBACK_CFG.get("enabled", True))
FEEDBACK_STORAGE_PATH = FEEDBACK_CFG.get(
    "storage_path", "diagnostics/user_feedback.jsonl"
)
FEEDBACK_ALLOWED_CATEGORIES = set(
    _normalize_keyword_list(FEEDBACK_CFG.get("categories"))
) or {"corretta", "incompleta", "fuori_fuoco", "troppo_formale", "troppo_lunga"}
ENTITY_STOPWORDS = {
    "con",
    "piu",
    "più",
    "piu'",
    "cliente",
    "qual",
    "quale",
    "chi",
    "cosa",
    "quanto",
}
CONTROL_FALLBACK_TOKENS = {
    "ricerca",
    "ricerche",
    "ambito",
    "ambiti",
    "perimetro",
    "perimetri",
    "scope",
    "analisi",
    "questa",
    "questo",
    "questa",
    "nuovo",
    "nuova",
    "tutti",
    "tutte",
    "la",
    "il",
    "lo",
    "un",
    "una",
    "gli",
    "le",
}

_CYPHER_WRITE_PATTERNS: List[Tuple[str, Pattern[str]]] = []
for kw in CYPHER_WRITE_BLOCKLIST:
    lowered = kw.lower()
    if not lowered:
        continue
    if any(ch in lowered for ch in (" ", ".", "_")):
        pattern = re.compile(re.escape(lowered), re.IGNORECASE)
    else:
        pattern = re.compile(rf"\b{re.escape(lowered)}\b", re.IGNORECASE)
    _CYPHER_WRITE_PATTERNS.append((kw, pattern))


def _is_small_talk(question: str) -> bool:
    """Riconosce scambi di cortesia o domande generiche da trattare come conversazione."""

    if not question:
        return False

    lowered = question.strip().lower()
    tokens = re.findall(r"\w+", lowered)

    if SMALL_TALK_KEYWORDS and _contains_keyword(lowered, tokens, SMALL_TALK_KEYWORDS):
        return True

    # fallback: frasi brevissime prive di cifre e con pochi token
    if len(tokens) <= 2 and not any(char.isdigit() for char in lowered):
        return True

    return False


def _contains_dangerous_intent(question: str) -> bool:
    """Intercetta richieste di inserimento/modifica/cancellazione da bloccare a monte."""

    if not question:
        return False

    lowered = question.strip().lower()
    tokens = re.findall(r"\w+", lowered)

    if DANGEROUS_KEYWORDS and _contains_keyword(lowered, tokens, DANGEROUS_KEYWORDS):
        return True

    return False


def _is_pure_rejection_reply(text: str) -> bool:
    """Riconosce risposte di controllo che indicano solo di cambiare ambito."""

    if not text:
        return True

    lowered = text.strip().lower()
    if not lowered:
        return True

    if len(lowered) > 80:
        return False

    tokens = re.findall(r"\w+", lowered)
    if not tokens:
        return True

    meaningful = []
    for token in tokens:
        if token in REJECTION_KEYWORDS:
            continue
        if token in CONTROL_FALLBACK_TOKENS:
            continue
        if len(token) <= 2:
            continue
        meaningful.append(token)

    return len(meaningful) == 0


def _sanitize_entity_value(value: Optional[str]) -> Optional[str]:
    """Normalizza e filtra valori poco significativi usati come anchor."""

    if value is None:
        return None
    raw = str(value).strip().strip("'\"")
    if not raw:
        return None
    val = raw.lower()
    if val in ENTITY_STOPWORDS:
        return None
    if len(val) < 3:
        return None
    return raw


def _detect_write_operation(query: Optional[str]) -> Optional[str]:
    """Ritorna la keyword vietata trovata nella query, se presente."""

    if not query:
        return None

    for keyword, pattern in _CYPHER_WRITE_PATTERNS:
        if pattern.search(query):
            return keyword
    return None


def _ensure_read_only_query(query: str) -> None:
    """Solleva un'eccezione se la query contiene operazioni di scrittura."""

    keyword = _detect_write_operation(query)
    if keyword:
        raise UnsafeCypherError(keyword)


def _classify_query_complexity(query: Optional[str]) -> Tuple[str, List[str]]:
    """Analizza la query e ritorna (livello, motivi)."""

    if not query:
        return "ok", []

    lowered = query.lower()
    reasons: List[str] = []

    # Pattern negati
    if re.search(
        r"where\s+not\s*\(\s*[a-z][a-z0-9_]*\s*-\s*\[", lowered, re.IGNORECASE
    ):
        reasons.append("negated_path")

    # MATCH senza label
    for match in re.finditer(r"match\s*\(\s*([a-z][a-z0-9_]*)\s*([^)]+)?\)", lowered):
        inner = match.group(0)
        if ":" not in inner:
            reasons.append("match_without_label")
            break

    # Assenza totale di WHERE e LIMIT
    if "return" in lowered and "where" not in lowered and "limit" not in lowered:
        reasons.append("missing_filters")
    elif "return" in lowered and "limit" not in lowered:
        reasons.append("no_limit")

    # Catene lunghe di relazioni
    chain_count = len(re.findall(r"-\[[^\]]*\]-", lowered))
    if chain_count >= 5:
        reasons.append("long_chain")

    if not reasons:
        return "ok", []

    severity_map = {
        "negated_path": "critical",
        "missing_filters": "critical",
        "long_chain": "risky",
        "match_without_label": "risky",
        "no_limit": "risky",
    }
    level = "risky"
    if any(severity_map.get(reason) == "critical" for reason in reasons):
        level = "critical"

    return level, sorted(set(reasons))


def _rewrite_query_safe(
    query: str, reasons: List[str]
) -> Tuple[Optional[str], List[str]]:
    """Prova a riscrivere la query in modo più sicuro. Restituisce (nuova_query, note)."""

    rewritten = query
    notes: List[str] = []
    changed = False

    if "negated_path" in reasons:
        new_query = _rewrite_negated_path_to_not_exists(rewritten)
        if new_query and new_query != rewritten:
            rewritten = new_query
            notes.append(
                "convertito WHERE NOT in NOT EXISTS con LIMIT {limit}".format(
                    limit=SAFE_REWRITE_LIMIT
                )
            )
            changed = True

    if not changed:
        return None, []
    return rewritten, notes


def _rewrite_negated_path_to_not_exists(query: str) -> Optional[str]:
    """Trasforma WHERE NOT (pattern) in WHERE NOT EXISTS { MATCH pattern LIMIT N }."""

    pattern = re.compile(r"where\s+not\s*\(", re.IGNORECASE)
    idx = 0
    result = query
    changed = False

    while True:
        match = pattern.search(result, idx)
        if not match:
            break

        segment = result[match.start() : match.start() + len("where not exists")]
        if segment.lower().startswith("where not exists"):
            idx = match.end()
            continue

        open_paren_idx = result.find("(", match.start())
        if open_paren_idx == -1:
            idx = match.end()
            continue

        depth = 1
        pos = open_paren_idx + 1
        while pos < len(result) and depth > 0:
            char = result[pos]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            pos += 1

        if depth != 0:
            idx = match.end()
            continue

        inner = result[open_paren_idx + 1 : pos - 1].strip()
        indent = _infer_indentation(result, match.start())
        replacement = (
            f"WHERE NOT EXISTS {{\n"
            f"{indent}    MATCH {inner}\n"
            f"{indent}    LIMIT {SAFE_REWRITE_LIMIT}\n"
            f"{indent}}}"
        )
        result = result[: match.start()] + replacement + result[pos:]
        idx = match.start() + len(replacement)
        changed = True

    return result if changed else None


def _infer_indentation(text: str, position: int) -> str:
    """Calcola l'indentazione (spazi) della linea corrente."""

    line_start = text.rfind("\n", 0, position)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line = text[line_start:position]
    indent = ""
    for ch in line:
        if ch in (" ", "\t"):
            indent += ch
        else:
            break
    return indent


def _log_slow_query(payload: Dict[str, Any]) -> None:
    """Append slow query diagnostics su file dedicato."""

    if not SLOW_QUERY_LOG_PATH:
        return
    try:
        path = Path(SLOW_QUERY_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Impossibile scrivere slow query log: %s", exc)


def _store_feedback_entry(entry: Dict[str, Any]) -> None:
    """Salva il feedback dell'utente in formato JSONL."""

    if not FEEDBACK_ENABLED:
        return
    try:
        path = Path(FEEDBACK_STORAGE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Impossibile salvare il feedback utente: %s", exc)


SCOPE_RESET_KEYWORDS = set(
    _normalize_keyword_list(MEMORY_CFG.get("scope_reset_keywords"))
)
EXPLICIT_RESET_KEYWORDS = set(
    _normalize_keyword_list(MEMORY_CFG.get("explicit_reset_keywords"))
)
FOLLOW_UP_KEYWORDS = set(_normalize_keyword_list(MEMORY_CFG.get("followup_keywords")))
PRONOUN_STARTS = tuple(_normalize_keyword_list(MEMORY_CFG.get("pronoun_starts")))
CONFIRMATION_KEYWORDS = set(_normalize_keyword_list(MEMORY_CFG.get("confirm_keywords")))
REJECTION_KEYWORDS = set(_normalize_keyword_list(MEMORY_CFG.get("reject_keywords")))

if not SCOPE_RESET_KEYWORDS:
    logger.warning(
        "sessions.memory.scope_reset_keywords non configurato: le domande generiche non resetteranno automaticamente lo scope."
    )
SHORT_QUESTION_TOKEN_THRESHOLD = int(
    MEMORY_CFG.get("short_question_token_threshold", 6) or 0
)
RECENCY_MINUTES_LIMIT = MEMORY_CFG.get("recency_minutes")
RECENCY_WINDOW = (
    timedelta(minutes=float(RECENCY_MINUTES_LIMIT))
    if RECENCY_MINUTES_LIMIT and float(RECENCY_MINUTES_LIMIT) > 0
    else None
)

EMBEDDING_CFG = MEMORY_CFG.get("embedding_similarity", {}) or {}
EMBEDDING_ACTIVE = bool(EMBEDDING_CFG.get("enabled", False))
EMBEDDING_HISTORY_SIZE = int(EMBEDDING_CFG.get("history_size", 3) or 0)
EMBEDDING_THRESHOLD = float(EMBEDDING_CFG.get("threshold", 0.75))
EMBEDDING_REUSE_EXAMPLES = bool(EMBEDDING_CFG.get("reuse_example_retriever", True))
EMBEDDING_MODEL_NAME = EMBEDDING_CFG.get("model_name", "all-MiniLM-L6-v2")
MEMORY_EMBEDDER = None

if EMBEDDING_ACTIVE and EMBEDDING_HISTORY_SIZE <= 0:
    logger.warning(
        "sessions.memory.embedding_similarity.history_size <= 0: la similarità embedding verrà ignorata."
    )

# Limiti query (da config/system.yaml)
MAX_RESULTS_FOR_SYNTHESIZER = config.system["query_limits"][
    "max_results_for_synthesizer"
]

logger.info(
    f"Cache configurata: enabled={ENABLE_CACHE}, max_size={config.system['cache']['max_cache_size']}, ttl={config.system['cache']['query_ttl_seconds']}s"
)
logger.info(f"Session timeout: {SESSION_TIMEOUT.total_seconds() / 60:.0f} minuti")
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


def _normalize_text_for_cache(text: str) -> str:
    """Normalizza testo per confronti cache (minuscolo, spazi puliti)."""

    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned


def _make_cache_key(user_id: str, original_question: str, normalized_task: str) -> str:
    """Crea una chiave cache che combina utente e domanda normalizzata."""

    base = _normalize_text_for_cache(normalized_task) or _normalize_text_for_cache(
        original_question
    )
    return f"{user_id}:::{base}"


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


def _compose_repair_task(original_question_it: str, english_task: str) -> str:
    """
    Prepara un blocco di testo che combina domanda originale e task inglese
    per aiutare l'agente di repair a conservare l'intento.
    """
    if not english_task:
        return original_question_it
    if not original_question_it:
        return english_task
    return (
        "Original user question (IT): "
        + original_question_it.strip()
        + "\nEnglish task: "
        + english_task.strip()
    )


def _error_matches_patterns(error_msg: str, patterns: List[str]) -> bool:
    """Verifica se il messaggio di errore combacia con i pattern configurati."""
    if not error_msg or not patterns:
        return False
    special_chars = set(".*?[]()^$|+{}\\")
    for pattern in patterns:
        if not pattern:
            continue
        is_regex = any(ch in special_chars for ch in pattern)
        try:
            if is_regex:
                if re.search(pattern, error_msg):
                    return True
            elif pattern in error_msg:
                return True
        except re.error:
            # In caso di regex mal formata ripiega sulla ricerca semplice
            if pattern in error_msg:
                return True
    return False


def _invoke_query_repair(
    *,
    english_task: str,
    original_question_it: str,
    relevant_schema: str,
    bad_query: str,
    error_msg: str,
    base_hints: str,
    recent_repairs: str,
) -> str:
    """Richiama l'agente LLM per correggere la query fallita."""
    if not QUERY_REPAIR_PROMPT:
        return bad_query

    task_block = _compose_repair_task(original_question_it, english_task)
    try:
        fixed = _invoke_with_profile(
            agent_name="coder",
            prompt_text=QUERY_REPAIR_PROMPT,
            variables={
                "question": task_block,
                "schema": relevant_schema,
                "bad_query": bad_query,
                "error": error_msg,
                "hints": base_hints,
                "recent_repairs": recent_repairs,
            },
        )
        candidate = extract_cypher(fixed)
        return candidate or bad_query
    except Exception as exc:
        logger.error(f"Errore durante il tentativo di query repair: {exc}")
        return bad_query


def _apply_preflight_sql_guard(
    *,
    current_query: str,
    english_task: str,
    original_question_it: str,
    relevant_schema: str,
    base_hints: str,
    recent_repairs: str,
) -> str:
    """Esegue un controllo preflight per intercettare sintassi SQL-like prima dell'esecuzione."""
    if not current_query or not QUERY_REPAIR_PROMPT:
        return current_query

    sql_like = re.compile(
        r"\b(GROUP\s+BY|HAVING|OVER\b|PARTITION\s+BY|ROW_NUMBER\s*\(|RANK\s*\(|WINDOW\b)",
        re.IGNORECASE,
    )
    if not sql_like.search(current_query or ""):
        return current_query

    logger.info(
        "Preflight Repair: rilevata sintassi SQL-like (GROUP BY/HAVING/OVER). Avvio correzione preventiva."
    )
    auto_hints = _suggest_unnecessary_with_hints(current_query)
    preflight_hints = (
        "SQL-like keywords are invalid in Cypher (GROUP BY, HAVING, OVER).\n"
        "Use Cypher aggregation with WITH/RETURN instead of SQL GROUP BY.\n"
        "Per ranking o finestre, usa WITH + collect(...) e ORDER BY, non funzioni OVER.\n"
    )
    hints_parts = [part for part in [base_hints, preflight_hints, auto_hints] if part]
    combined_hints = "\n\n".join(hints_parts).strip()
    fixed_query = _invoke_query_repair(
        english_task=english_task,
        original_question_it=original_question_it,
        relevant_schema=relevant_schema,
        bad_query=current_query,
        error_msg="Detected SQL-like syntax (GROUP BY/HAVING/OVER) not valid in Cypher.",
        base_hints=combined_hints,
        recent_repairs=recent_repairs,
    )
    if fixed_query and fixed_query != current_query:
        logger.info(f"Query corretta (preflight) proposta:\n{fixed_query}")
        _append_repair_event(
            english_task,
            current_query,
            "Preflight SQL-like syntax",
            fixed_query,
        )
        return fixed_query
    return current_query


def execute_query_with_iterative_improvement(
    *,
    initial_query: str,
    english_task: str,
    original_question_it: str,
    relevant_schema: str,
    retry_cfg: Dict[str, Any],
    base_hints_text: str,
    recent_repairs: str,
    semantic_cfg: Optional[Dict[str, Any]] = None,
    semantic_examples: Optional[List[Dict[str, Any]]] = None,
    contextualized_question: Optional[str] = None,
    execution_timeout: Optional[float] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Esegue la query su Neo4j provando a correggerla iterativamente.

    Ritorna la query finale, il contesto completo e un audit trail dei tentativi.
    """
    attempts = max(1, int(retry_cfg.get("max_attempts", 1)))
    patterns = retry_cfg.get("repairable_error_patterns", [])
    current_query = initial_query
    audit_trail: List[Dict[str, str]] = []
    last_error_msg: Optional[str] = None

    semantic_cfg = semantic_cfg or {}
    semantic_enabled = bool(semantic_cfg.get("enabled", False)) and bool(
        QUERY_EXPANSION_PROMPT
    )
    semantic_max_attempts = int(
        semantic_cfg.get("max_attempts", SEMANTIC_EXPANSION_MAX_ATTEMPTS_DEFAULT) or 0
    )
    semantic_top_k = int(
        semantic_cfg.get("top_k_examples", SEMANTIC_EXPANSION_TOP_K_DEFAULT) or 0
    )
    semantic_attempts = 0

    # Preflight per sintassi SQL
    current_query = _apply_preflight_sql_guard(
        current_query=current_query,
        english_task=english_task,
        original_question_it=original_question_it,
        relevant_schema=relevant_schema,
        base_hints=base_hints_text,
        recent_repairs=recent_repairs,
    )
    _ensure_read_only_query(current_query)

    for attempt_idx in range(1, attempts + 1):
        try:
            logger.info(f"Esecuzione attempt {attempt_idx}/{attempts} su Neo4j.")
            _ensure_read_only_query(current_query)
            context = execute_cypher_with_timeout(
                current_query, timeout_seconds=execution_timeout
            )
            return current_query, context, audit_trail
        except QueryTimeoutError as timeout_exc:
            last_error_msg = str(timeout_exc)
            logger.error(
                f"Tentativo {attempt_idx}/{attempts} fallito per timeout: {last_error_msg}"
            )
            raise
        except Exception as exc:
            error_msg = str(exc)
            last_error_msg = error_msg
            logger.warning(
                f"Tentativo {attempt_idx}/{attempts} fallito con errore: {error_msg[:200]}"
            )

            audit_entry = {
                "attempt": attempt_idx,
                "bad_query": current_query,
                "error": error_msg,
            }

            # Ultimo tentativo: uscita
            if attempt_idx >= attempts:
                audit_trail.append(audit_entry)
                break

            if not QUERY_REPAIR_PROMPT:
                audit_trail.append(audit_entry)
                break

            if not _error_matches_patterns(error_msg, patterns):
                audit_trail.append(audit_entry)
                break

            auto_hints = _suggest_unnecessary_with_hints(current_query)
            hints_parts = [part for part in [base_hints_text, auto_hints] if part]
            combined_hints = "\n\n".join(hints_parts).strip()

            fixed_query = _invoke_query_repair(
                english_task=english_task,
                original_question_it=original_question_it,
                relevant_schema=relevant_schema,
                bad_query=current_query,
                error_msg=error_msg,
                base_hints=combined_hints,
                recent_repairs=recent_repairs,
            )
            audit_entry["fixed_query"] = fixed_query
            audit_trail.append(audit_entry)

            if not fixed_query or fixed_query.strip() == current_query.strip():
                logger.info("Repair agent non ha proposto cambiamenti significativi.")
                break

            logger.info(f"Query corretta proposta:\n{fixed_query}")
            _append_repair_event(
                english_task,
                current_query,
                error_msg,
                fixed_query,
            )
            _ensure_read_only_query(fixed_query)
            current_query = fixed_query

    # Tentativo finale di espansione semantica se abilitato
    if semantic_enabled and semantic_max_attempts > 0:
        semantic_examples = semantic_examples or []
        if semantic_top_k > 0:
            semantic_examples = semantic_examples[:semantic_top_k]
        semantic_attempts = 0
        while semantic_attempts < semantic_max_attempts:
            semantic_query = _semantic_expansion_attempt(
                question_it=original_question_it,
                contextualized_question=contextualized_question,
                english_task=english_task,
                relevant_schema=relevant_schema,
                bad_query=current_query,
                error_msg=last_error_msg or "",
                semantic_examples=semantic_examples,
                base_hints=base_hints_text,
                recent_repairs=recent_repairs,
                top_k=semantic_top_k,
            )

            semantic_attempts += 1
            if not semantic_query or semantic_query.strip() == current_query.strip():
                logger.info("Semantic expansion non ha prodotto una nuova query utile.")
                continue

            audit_entry = {
                "attempt": f"semantic-{semantic_attempts}",
                "bad_query": current_query,
                "error": last_error_msg or "",
                "semantic_expansion": True,
                "fixed_query": semantic_query,
            }

            try:
                logger.info("Esecuzione query dopo espansione semantica.")
                _ensure_read_only_query(semantic_query)
                context = execute_cypher_with_timeout(
                    semantic_query, timeout_seconds=execution_timeout
                )
                audit_trail.append(audit_entry)
                return semantic_query, context, audit_trail
            except Exception as semantic_exc:
                last_error_msg = str(semantic_exc)
                audit_entry["semantic_error"] = last_error_msg
                audit_trail.append(audit_entry)
                current_query = semantic_query
                _ensure_read_only_query(current_query)

    # se siamo qui tutti i tentativi sono falliti
    raise Exception(last_error_msg or "Query execution failed without error message.")


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


class UnsafeCypherError(RuntimeError):
    """Raised when a generated Cypher query contiene operazioni di scrittura non permesse."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        super().__init__(f"Operazione Cypher non consentita rilevata: {keyword}")


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
    query: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
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


def execute_cypher_with_records(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
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
            return list(result)
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
    meta: Dict[str, Any] = Field(default_factory=dict)


class ConversationSession(BaseModel):
    user_id: str
    messages: List[ConversationMessage] = []
    created_at: datetime
    last_activity: datetime


class FeedbackPayload(BaseModel):
    user_id: str
    question: str
    category: str
    notes: Optional[str] = None
    answer: Optional[str] = None
    query_generated: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Cache globale per le sessioni di conversazione (scade dopo timeout configurato)
conversation_sessions: Dict[str, ConversationSession] = {}

# Controlla quante interazioni recenti passiamo al contextualizer
MAX_HISTORY_FOR_CONTEXTUALIZER = 4
# Lunghezza massima (caratteri) per sintesi testo memorizzata nel prompt
MAX_CONTEXT_SNIPPET_LENGTH = 220


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
    user_id: str,
    question: str,
    answer: str,
    context: List[Dict],
    query: str,
    meta: Optional[Dict[str, Any]] = None,
):
    """Aggiungi un messaggio alla sessione di conversazione"""
    if not ENABLE_MEMORY:
        return

    session = get_or_create_session(user_id)

    message = ConversationMessage(
        timestamp=datetime.now(),
        question=question,
        answer=answer,
        context=context,
        query_generated=query,
        meta=meta or {},
    )

    session.messages.append(message)

    # Mantengo solo gli ultimi 5 messaggi per evitare memory overflow
    if len(session.messages) > MAX_MESSAGES_PER_SESSION:
        session.messages = session.messages[-MAX_MESSAGES_PER_SESSION:]

    logger.info(
        f" Messaggio aggiunto alla sessione. Totale messaggi: {len(session.messages)}"
    )


def _summarize_for_memory(
    text: str, max_length: int = MAX_CONTEXT_SNIPPET_LENGTH
) -> str:
    """Compatta un testo rimuovendo spazi ripetuti e tronca in modo sicuro."""

    if not text:
        return ""
    sanitized = re.sub(r"\s+", " ", text.strip())
    if len(sanitized) <= max_length:
        return sanitized
    return sanitized[: max_length - 3].rstrip() + "..."


def _collect_entities_from_context(context: List[Dict]) -> Dict[str, set]:
    """Estrae rapidamente entità note dal contesto della risposta precedente."""

    buckets = {
        "clienti": set(),
        "fornitori": set(),
        "ditte": set(),
        "prodotti": set(),
        "famiglia": set(),
        "documenti": set(),
    }

    for item in context or []:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if value in (None, ""):
                continue
            lower_key = str(key).lower()
            # Normalizzo il valore a stringa per facilitarne il prompt engineering
            val_str = str(value)
            if "fornit" in lower_key:
                buckets["fornitori"].add(val_str)
            elif "ditta" in lower_key:
                buckets["ditte"].add(val_str)
            elif "articolo" in lower_key or "prodotto" in lower_key:
                buckets["prodotti"].add(val_str)
            elif "cliente" in lower_key:
                buckets["clienti"].add(val_str)
            elif "famiglia" in lower_key:
                buckets["famiglia"].add(val_str)
            elif "documento" in lower_key or "doc_" in lower_key:
                buckets["documenti"].add(val_str)
    return buckets


def _question_mentions_known_entities(
    question: str, session: ConversationSession
) -> bool:
    """Verifica se la domanda fa riferimento a entità note nelle interazioni precedenti."""
    if not question or not session.messages:
        return False

    lowered_question = question.lower()
    known_entities: set = set()
    recent_messages = session.messages[-MAX_HISTORY_FOR_CONTEXTUALIZER:]
    for message in recent_messages:
        entities = _collect_entities_from_context(message.context)
        for values in entities.values():
            for val in values:
                if not val:
                    continue
                known_entities.add(str(val).lower())
        explicit = _extract_explicit_identifiers(message.question)
        for values in explicit.values():
            for val in values:
                if not val:
                    continue
                known_entities.add(str(val).lower())
    return any(entity and entity in lowered_question for entity in known_entities)


def _should_use_memory(question: str, session: ConversationSession) -> bool:
    """
    Stabilisce se includere la cronologia. Versione "intelligente" (pulita).
    Si fida del Contextualizer per capire i follow-up.
    """
    if not question:
        return False

    # 1. CONTROLLO "SMALL TALK" (Fix per "Grazie mille")
    if _is_small_talk(question):
        logger.info("Memory routing: Rilevato small talk. Nessuna memoria usata.")
        return False

    # 2. CONTROLLO "PRIMA DOMANDA"
    if not session.messages:
        logger.info("Memory routing: Prima domanda, nessuna memoria usata.")
        return False

    # 3. CONTROLLO "RESET ESPLICITO" (l'utente lo chiede)
    lowered = question.strip().lower()
    tokens = re.findall(r"\w+", lowered)
    if _contains_keyword(lowered, tokens, EXPLICIT_RESET_KEYWORDS):
        logger.info("Memory reset: Rilevato reset esplicito.")
        session.messages = []  # Svuota la memoria
        return False

    # 4. CONTROLLO "RESET STRUTTURALE" (Fix per "Lombardia")
    if _should_reset_scope(question):  # Chiama la funzione "intelligente"
        logger.info("Memory reset: Rilevata domanda autosufficiente.")
        return False

    # Se non è niente di tutto ciò, È UN FOLLOW-UP.
    # Passiamo la palla al Contextualizer per decidere COME usarla.
    logger.info(
        "Memory routing: Domanda non-reset, passo al Contextualizer con cronologia."
    )
    return True


def _max_embedding_similarity(
    question: str, session: ConversationSession
) -> Optional[float]:
    """Calcola la massima similarità coseno tra la domanda attuale e la cronologia recente."""
    if not EMBEDDING_ACTIVE or MEMORY_EMBEDDER is None or EMBEDDING_HISTORY_SIZE <= 0:
        return None
    try:
        history_questions = [
            msg.question
            for msg in session.messages[-EMBEDDING_HISTORY_SIZE:]
            if msg.question
        ]
        if not history_questions:
            return None

        texts = [question] + history_questions
        embeddings = MEMORY_EMBEDDER.encode(texts, show_progress_bar=False)
        if embeddings is None or len(embeddings) <= 1:
            return None

        query_vec = np.array(embeddings[0], dtype=np.float32)
        history_vecs = np.array(embeddings[1:], dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        history_norms = np.linalg.norm(history_vecs, axis=1)

        if query_norm == 0 or np.any(history_norms == 0):
            return None

        sims = (history_vecs @ query_vec) / (history_norms * query_norm)
        if sims.size == 0:
            return None
        return float(np.max(sims))
    except Exception as exc:
        logger.warning(f"Similarity embedding fallita: {exc}")
        return None


def _format_semantic_examples_for_prompt(
    examples: Optional[List[Dict[str, Any]]], top_k: int
) -> str:
    if not examples:
        return ""
    lines: List[str] = []
    limit = top_k if top_k > 0 else len(examples)
    for idx, ex in enumerate(examples[:limit], start=1):
        question = ex.get("question", "").strip()
        sim = ex.get("similarity")
        header = f"{idx}. {question}" if question else f"{idx}."
        if isinstance(sim, (int, float)):
            header += f" (sim {float(sim):.2f})"
        lines.append(header)
        cypher = (ex.get("cypher") or "").strip()
        if cypher:
            first_line = cypher.splitlines()[0]
            lines.append(f"   Cypher: {first_line}")
    return "\n".join(lines)


def _semantic_expansion_attempt(
    *,
    question_it: str,
    contextualized_question: Optional[str],
    english_task: str,
    relevant_schema: str,
    bad_query: str,
    error_msg: str,
    semantic_examples: Optional[List[Dict[str, Any]]],
    base_hints: str,
    recent_repairs: str,
    top_k: int,
) -> str:
    if not SEMANTIC_EXPANSION_ENABLED or not QUERY_EXPANSION_PROMPT:
        return ""

    examples_block = _format_semantic_examples_for_prompt(semantic_examples, top_k)
    variables = {
        "question_it": contextualized_question or question_it,
        "english_task": english_task,
        "schema": relevant_schema,
        "bad_query": bad_query,
        "error": error_msg,
        "examples": examples_block or "",
        "recent_repairs": recent_repairs or "",
        "hints": base_hints or "",
    }
    try:
        expanded = _invoke_with_profile(
            agent_name="coder",
            prompt_text=QUERY_EXPANSION_PROMPT,
            variables=variables,
        )
        return extract_cypher(expanded)
    except Exception as exc:
        logger.error(f"Errore durante la semantic expansion: {exc}")
        return ""


def _extract_explicit_identifiers(text: str) -> Dict[str, set]:
    """Trova riferimenti espliciti (es. ditta '1') nella domanda originale."""

    ids = {
        "ditta": set(),
        "cliente": set(),
        "fornitore": set(),
    }
    if not text:
        return ids

    patterns = {
        "ditta": r"ditta\s*['\"]?(\w+)['\"]?",
        "cliente": r"cliente\s*['\"]?(\w+)['\"]?",
        "fornitore": r"fornitore\s*['\"]?(\w+)['\"]?",
    }

    lowered = text.lower()
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            ids[label].add(match.group(1))
    return ids


def _should_reset_scope(question: str) -> bool:
    """
    Determina se la nuova domanda richiede di ignorare lo scope precedente.
    """

    if not question:
        return False

    lowered = question.lower().strip()
    tokens = set(re.findall(r"\w+", lowered))

    # 1. Controllo keyword esplicite da config (es. "nuova ricerca")
    if _contains_keyword(lowered, list(tokens), EXPLICIT_RESET_KEYWORDS):
        logger.info("Memory reset: Rilevata keyword di reset esplicito.")
        return True

    # 2. Controllo keyword generiche da config (es. "statistiche")
    if _contains_keyword(lowered, list(tokens), SCOPE_RESET_KEYWORDS):
        logger.info("Memory reset: Rilevata keyword di reset generico.")
        return True

    # 3. Controllo STRUTTURALE (La vera soluzione)
    #    Se la domanda è una NUOVA interrogativa autosufficiente, resetta.
    interrogative_list = MEMORY_CFG.get("INTERROGATIVE_START")
    dominio_list = MEMORY_CFG.get("DOMINIO_KEYWORDS")
    dominio_set = set(dominio_list) if dominio_list else set()
    # Controlla se esiste e convertila in TUPLA
    if interrogative_list and lowered.startswith(tuple(interrogative_list)):
        # a) Contiene parole chiave del DOMINIO (es. "Qual è il fatturato...")
        if dominio_set.intersection(tokens):
            logger.info("Memory reset: Rilevata domanda 'Wh-' con keyword di dominio.")
            return True

        # b) Contiene un NOME PROPRIO (euristica generale per nomi di persone/aziende)
        original_tokens = re.findall(r"\w+", question)
        if len(original_tokens) > 1:
            if any(t[0].isupper() for t in original_tokens[1:]):
                logger.info(
                    "Memory reset: Rilevata domanda 'Wh-' con probabile Nome Proprio."
                )
                return True

    return False


def _scope_summary_for_synth(original: str, contextualized: str) -> str:
    """Genera un promemoria sintetico sui filtri interpretati dal contextualizer."""

    if not contextualized:
        return ""

    norm_original = _normalize_text_for_cache(original)
    norm_contextualized = _normalize_text_for_cache(contextualized)
    if norm_original == norm_contextualized:
        return ""

    explicit_refs = _extract_explicit_identifiers(contextualized)
    fragments: List[str] = []

    if explicit_refs.get("ditta"):
        ditte = ", ".join(sorted({f"ditta '{val}'" for val in explicit_refs["ditta"]}))
        fragments.append(f"clienti appartenenti a {ditte}")
    if explicit_refs.get("cliente"):
        clienti = ", ".join(
            sorted({f"cliente '{val}'" for val in explicit_refs["cliente"]})
        )
        fragments.append(f"le entità {clienti}")
    if explicit_refs.get("fornitore"):
        fornitori = ", ".join(
            sorted({f"fornitore '{val}'" for val in explicit_refs["fornitore"]})
        )
        fragments.append(f"i dati relativi a {fornitori}")

    if fragments:
        detail = "; ".join(fragments)
        templates = [
            "Perimetro considerato: {detail}.",
            "Ambito di analisi corrente: {detail}.",
            "Sto focalizzando la risposta su {detail}.",
            "Analisi limitata a {detail}.",
        ]
        digest = hashlib.md5(norm_contextualized.encode("utf-8")).hexdigest()
        idx = int(digest[:2], 16) % len(templates)
        return templates[idx].format(detail=detail)

    return f'Interpretazione contestualizzata: "{contextualized}"'


def get_conversation_context(
    user_id: str, incoming_question: Optional[str] = None
) -> str:
    """Costruisce un contesto multi-turno compatto per il contextualizer."""

    if not ENABLE_MEMORY:
        return ""

    session = conversation_sessions.get(user_id)
    if not session or not session.messages:
        return ""

    if incoming_question and not _should_use_memory(incoming_question, session):
        logger.info(
            "Memory routing: _should_use_memory ha dato False. Cronologia esclusa."
        )
        return ""
    # --- FINE MODIFICA ---

    history_slice = session.messages[-MAX_HISTORY_FOR_CONTEXTUALIZER:]
    context_lines: List[str] = ["CONVERSAZIONE PRECEDENTE (più recente per primo):"]

    aggregated_entities = {
        "clienti": set(),
        "fornitori": set(),
        "ditte": set(),
        "prodotti": set(),
        "famiglia": set(),
        "documenti": set(),
    }
    explicit_refs = {"ditta": set(), "cliente": set(), "fornitore": set()}

    for offset, message in enumerate(reversed(history_slice), start=1):
        context_lines.append(f"[Turno -{offset}]")
        context_lines.append(f"Domanda: {message.question}")

        explicit = _extract_explicit_identifiers(message.question)
        for label, values in explicit.items():
            explicit_refs[label].update(values)

        if message.query_generated:
            context_lines.append(
                "QueryCypher: " + _summarize_for_memory(message.query_generated, 260)
            )

        if message.answer:
            context_lines.append("Risposta: " + _summarize_for_memory(message.answer))

        entities = _collect_entities_from_context(message.context)
        for bucket, values in entities.items():
            if bucket not in aggregated_entities:
                aggregated_entities[bucket] = set()
            aggregated_entities[bucket].update(values)

    # Inserisco blocchi riassuntivi finali per aiutare il contextualizer a non perdere riferimenti
    entity_block = []
    for label, values in aggregated_entities.items():
        if values:
            entity_block.append(f"{label.upper()} = {sorted(values)}")

    if entity_block:
        context_lines.append("ENTITA_RILEVANTI:")
        context_lines.extend(entity_block)

    explicit_block = []
    for label, values in explicit_refs.items():
        if values:
            explicit_block.append(
                f"RIFERIMENTI_ESPLICITI_{label.upper()} = {sorted(values)}"
            )
    if explicit_block:
        context_lines.append(
            "NOTA: Mantieni invariati i riferimenti testuali espliciti (es. ID)."
        )
        context_lines.extend(explicit_block)

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
    token_pattern = re.compile(
        r"""(?:"[^"]+"|'[^']+'|“[^”]+”|‘[^’]+’|`[^`]+`|«[^»]+»|„[^“]+“|\S+)"""
    )
    raw_tokens = token_pattern.findall(question)
    parole: List[str] = []
    token_map: Dict[str, str] = {}
    for token in raw_tokens:
        cleaned = token.strip("'\"“”‘’`”„«»")
        cleaned = re.sub(r"^[,.;:!?]+|[,.;:!?]+$", "", cleaned)
        if cleaned and len(cleaned) >= MIN_WORD_LENGTH:
            parole.append(cleaned)
            token_map[cleaned] = token

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
                original_token = token_map.get(parola, parola)
                prefix = ""
                suffix = ""
                if original_token and original_token[0] in "'\"“”‘’`”„«»":
                    prefix = original_token[0]
                if original_token and original_token[-1] in "'\"“”‘’`”„«»":
                    suffix = original_token[-1]
                wrapped_best_match = f"{prefix}{best_match}{suffix}"
                if original_token:
                    question = question.replace(original_token, wrapped_best_match, 1)
                # Sostituisci eventuali occorrenze nude della parola
                if original_token != parola and re.search(
                    rf"\b{re.escape(parola)}\b", question
                ):
                    question = re.sub(
                        rf"\b{re.escape(parola)}\b",
                        wrapped_best_match,
                        question,
                        count=1,
                    )
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
QUERY_EXPANSION_PROMPT = (
    load_prompt_from_yaml(resolve_prompt_path("coder", SEMANTIC_EXPANSION_PROMPT_NAME))
    if SEMANTIC_EXPANSION_ENABLED
    else None
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
required_prompts = [
    BASE_CYPHER_RULES,
    SPECIALIST_ROUTER_PROMPT,
    SPECIALIST_CODERS,
    ADVANCED_CONTEXTUALIZER_PROMPT,
    SYNTHESIZER_PROMPT,
    TRANSLATOR_PROMPT,
    GENERAL_CONVERSATION_PROMPT,
    QUERY_REPAIR_PROMPT,
]
if SEMANTIC_EXPANSION_ENABLED:
    required_prompts.append(QUERY_EXPANSION_PROMPT)

if not all(required_prompts):
    print("Uno o più file prompt non sono stati caricati. Il programma terminerà.")
    exit()

logger.info("📚 Inizializzo Example Retriever...")
from example_retriever import ExampleRetriever

example_retriever = ExampleRetriever(min_similarity=EXAMPLES_MIN_SIMILARITY)
logger.info(f"✅ Esempi: {example_retriever.get_stats()}")

if EMBEDDING_ACTIVE:
    try:
        if EMBEDDING_REUSE_EXAMPLES and hasattr(example_retriever, "model"):
            MEMORY_EMBEDDER = example_retriever.model
        if MEMORY_EMBEDDER is None:
            from sentence_transformers import SentenceTransformer

            MEMORY_EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if MEMORY_EMBEDDER is not None:
            logger.info(
                f"Memoria sessione: embedding attivo (soglia {EMBEDDING_THRESHOLD:.2f}, history {EMBEDDING_HISTORY_SIZE})."
            )
        else:
            EMBEDDING_ACTIVE = False
            logger.warning(
                "Memoria sessione: embedding disattivato (modello non disponibile)."
            )
    except Exception as exc:
        EMBEDDING_ACTIVE = False
        MEMORY_EMBEDDER = None
        logger.warning(
            f"Impossibile inizializzare il modello di embedding per la memoria: {exc}"
        )


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
    examples_top_k: Optional[int] = None,
) -> Tuple[str, List[Dict], List[Dict], Optional[float]]:
    """Coder con esempi dinamici RAG"""
    logger.info(f"🤖 Coder: {specialist_type}")

    # 1. RECUPERA ESEMPI
    # Usa la domanda originale in IT per il retrieval degli esempi
    top_k_examples = (
        examples_top_k
        if examples_top_k is not None
        else example_retriever.get_default_top_k()
    )
    raw_examples = example_retriever.retrieve(
        question=original_question_it,
        specialist=specialist_type,
        top_k=top_k_examples,
        allow_low_similarity=True,
    )

    best_similarity = example_retriever.last_similarity
    use_examples = raw_examples
    min_sim = getattr(example_retriever, "min_similarity", None)
    if (
        min_sim is not None
        and best_similarity is not None
        and best_similarity < float(min_sim)
    ):
        logger.info(
            "📏 Similarità esempi %.3f sotto soglia %.3f: lo specialista %s procederà senza esempi RAG.",
            best_similarity,
            float(min_sim),
            specialist_type,
        )
        use_examples = []
    elif best_similarity is not None:
        logger.info(
            "📏 Similarità esempi selezionati: %.3f (threshold=%s)",
            best_similarity,
            f"{float(min_sim):.2f}" if min_sim is not None else "n/a",
        )
    else:
        logger.info("📏 Nessun esempio recuperato o similarità non disponibile.")

    # 2. FORMATTA ESEMPI (con escape delle graffe)
    if use_examples:
        examples_text = "\n\n**RELEVANT EXAMPLES:**\n" + "\n\n".join(
            [
                f"Q: {ex['question']}\n```cypher\n{ex['cypher'].strip().replace('{', '{{').replace('}', '}}')}\n```"
                for ex in use_examples
            ]
        )
        logger.info(
            f"📚 Recuperati (top_k={top_k_examples}): {[ex['id'] for ex in use_examples]}"
        )
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

    return extract_cypher(query), use_examples, raw_examples, best_similarity


def run_contextualizer_agent(question: str, chat_history: str) -> str:
    """
    Riscrive una domanda di follow-up in una domanda autonoma, gestendo la memoria.
    """
    reset_scope = _should_reset_scope(question)
    if reset_scope:
        logger.info("Contextualizer: il testo sembra richiedere uno scope generale.")

    if not ENABLE_MEMORY or not chat_history:
        return question  # Se non c'è cronologia, restituisce la domanda originale

    logger.info("Chiamata al Contextualizer Avanzato...")
    rewritten_question = _invoke_with_profile(
        agent_name="contextualizer",
        prompt_text=ADVANCED_CONTEXTUALIZER_PROMPT,
        variables={
            "chat_history": chat_history,
            "question": question,
            "scope_hint": "RESET_SCOPE" if reset_scope else "",
        },
    )
    return rewritten_question.strip()


def run_synthesizer_agent(
    question: str,
    context_str: str,
    total_results: int,
    contextualized_question: Optional[str] = None,
    filters_summary: Optional[str] = None,
) -> str:
    logger.info("Chiamata al Synthesizer...")

    contextualized_question = contextualized_question or question
    answer = _invoke_with_profile(
        agent_name="synthesizer",
        prompt_text=SYNTHESIZER_PROMPT,
        variables={
            "question": question,
            "contextualized_question": contextualized_question,
            "context": context_str,
            "total_results": total_results,
            "filters_summary": filters_summary or "",
        },
    )
    return answer.strip()


# FUNZIONI AUSILIARIE VARIE
def _extract_node_identity(node: Any) -> Optional[str]:
    """Estrae l'ID univoco del nodo/relazione (compatibile v4/v5)."""
    if node is None:
        return None

    # Neo4j v5+ usa element_id (stringa)
    if hasattr(node, "element_id"):
        try:
            return str(getattr(node, "element_id"))
        except Exception:
            pass  # Continua con i fallback

    # Fallback per Neo4j v4 (int)
    for attr in ("id", "identity"):
        if hasattr(node, attr):
            try:
                # Converte sempre in stringa per coerenza
                return str(int(getattr(node, attr)))
            except Exception:
                continue

    logger.debug("Impossibile estrarre un ID per il nodo/relazione")
    return None


def _node_to_payload(node: Node) -> Dict[str, Any]:
    node_id = _extract_node_identity(node)  # Ora ritorna una stringa (o None)
    labels = list(node.labels) if hasattr(node, "labels") else []
    properties = {k: make_context_json_serializable(v) for k, v in node.items()}
    return {
        "id": node_id,  # Già stringa (o None)
        "labels": labels,
        "properties": properties,
    }


def _relationship_to_payload(rel: Relationship) -> Dict[str, Any]:
    rel_id = _extract_node_identity(rel)  # Stringa

    start_id = None
    # Neo4j v5+ ha gli ID direttamente
    if hasattr(rel, "start_node_element_id"):
        start_id = str(rel.start_node_element_id)
    # Fallback per v4
    elif hasattr(rel, "start_node"):
        start_id = _extract_node_identity(rel.start_node)

    end_id = None
    # Neo4j v5+ ha gli ID direttamente
    if hasattr(rel, "end_node_element_id"):
        end_id = str(rel.end_node_element_id)
    # Fallback per v4
    elif hasattr(rel, "end_node"):
        end_id = _extract_node_identity(rel.end_node)

    properties = {k: make_context_json_serializable(v) for k, v in rel.items()}
    return {
        "id": rel_id,  # Già stringa
        "type": getattr(rel, "type", None),
        "source": start_id,  # Già stringa
        "target": end_id,  # Già stringa
        "properties": properties,
    }


def extract_graph_snapshot(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Estrae nodi e relazioni dai risultati Neo4j per visualizzazione front-end.
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}

    def visit(value: Any):
        """Visita ricorsivamente i valori per estrarre nodi e relazioni."""
        if value is None:
            return

        # Gestione nodi
        if isinstance(value, Node):
            node_id = _extract_node_identity(value)
            if node_id:
                key = str(node_id)
                if key not in nodes:
                    nodes[key] = _node_to_payload(value)

        # Gestione relazioni
        elif isinstance(value, Relationship):
            payload = _relationship_to_payload(value)
            edge_id = payload.get("id")
            if edge_id:
                edges[str(edge_id)] = payload

            # Visita anche i nodi di inizio e fine
            if hasattr(value, "start_node") and value.start_node:
                visit(value.start_node)
            if hasattr(value, "end_node") and value.end_node:
                visit(value.end_node)

        # Gestione path
        elif isinstance(value, Path):
            for node in value.nodes:
                visit(node)
            for rel in value.relationships:
                visit(rel)

        # Gestione collezioni
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

        # Gestione dizionari
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    # Processa tutti i records
    for record in records or []:
        if isinstance(record, dict):
            for value in record.values():
                visit(value)
        else:
            # Records potrebbero essere oggetti Record di Neo4j
            try:
                for value in record.values():
                    visit(value)
            except Exception as e:
                logger.debug(f"Errore processando record: {e}")

    # Verifica se abbiamo trovato qualcosa
    if not nodes and not edges:
        logger.debug("Nessun nodo o relazione trovata nei records")
        return {}

    # Limiti di sicurezza
    max_nodes = 200
    max_edges = 400

    if len(nodes) > max_nodes or len(edges) > max_edges:
        logger.warning(
            f"Grafo troppo grande (nodi={len(nodes)}, relazioni={len(edges)}). "
            f"Limiti: max_nodes={max_nodes}, max_edges={max_edges}"
        )
        return {}

    logger.info(f"Grafo estratto: {len(nodes)} nodi, {len(edges)} relazioni")
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


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
    if isinstance(context, Node):
        payload = _node_to_payload(context)
        payload["_type"] = "node"
        return payload
    if isinstance(context, Relationship):
        payload = _relationship_to_payload(context)
        payload["_type"] = "relationship"
        return payload
    if isinstance(context, Path):
        return {
            "_type": "path",
            "nodes": [make_context_json_serializable(node) for node in context.nodes],
            "relationships": [
                make_context_json_serializable(rel) for rel in context.relationships
            ],
        }
    # Gestisce tutti i tipi di dato temporale di Neo4j e Python
    if isinstance(context, (date, Date, DateTime, Time, Duration)):
        return str(context)  # La conversione a stringa è universale e sicura
    return context


def _extract_variables_from_query(query_prefix: str) -> Tuple[List[str], List[str]]:
    """
    Estrae tutti i nomi di variabili di nodi E RELAZIONI
    da una stringa di query Cypher (prefisso MATCH/WHERE/WITH).
    """
    # Regex per nodi: (varName:Label) o (varName)
    # \(([a-zA-Z0-9_]+) -> cattura il nome della variabile (es. 'c')
    # \s*(?::[a-zA-Z0-9_`| ]+)? -> opzionalmente matcha :Label, :`Label con Spazi`, :Label1|Label2
    node_pattern = re.compile(r"\(([a-zA-Z0-9_]+)\s*(?::[a-zA-Z0-9_`| ]+)?\s*\)")

    # Regex per relazioni: -[varName:TYPE]- o -[varName]-
    # -\[([a-zA-Z0-9_]+) -> cattura il nome della variabile (es. 'r')
    # \s*(?::[a-zA-Z0-9_`|* ]+)? -> opzionalmente matcha :TYPE, :`TYPE`, :TYPE1|TYPE2, :*
    rel_pattern = re.compile(r"-\[([a-zA-Z0-9_]+)\s*(?::[a-zA-Z0-9_`|* ]+)?\s*\]-")

    node_vars: List[str] = []
    rel_vars: List[str] = []
    node_seen: Set[str] = set()
    rel_seen: Set[str] = set()

    # Trova tutti i nodi
    for match in node_pattern.finditer(query_prefix):
        var_name = match.group(1)
        if var_name:  # Ignora variabili anonime come ()
            if var_name not in node_seen:
                node_seen.add(var_name)
                node_vars.append(var_name)

    # Trova tutte le relazioni
    for match in rel_pattern.finditer(query_prefix):
        var_name = match.group(1)
        if var_name:  # Ignora variabili anonime come -[]-
            if var_name not in rel_seen:
                rel_seen.add(var_name)
                rel_vars.append(var_name)

    if not node_vars and not rel_vars:
        logger.debug("Nessuna variabile trovata nel prefisso query.")

    return node_vars, rel_vars


def _extract_missing_variable_from_error(message: str) -> Optional[str]:
    """Prova a estrarre il nome della variabile assente dal messaggio di errore Neo4j."""

    if not message:
        return None
    match = re.search(r"Variable\s+`([^`]+)`\s+not defined", message)
    if match:
        return match.group(1)
    match = re.search(r"Variable\s+'([^']+)'\s+not defined", message)
    if match:
        return match.group(1)
    return None


def _reconstruct_graph_from_query(query: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Ricostruisce un grafo da query che restituiscono aggregazioni (COUNT, SUM, etc.).
    STRATEGIA: Assegna variabili a tutti i nodi anonimi, poi estrae tutto.
    """
    try:
        if not query:
            return {}

        query_clean = query.strip()

        # Trova l'ULTIMO RETURN
        return_matches = list(re.finditer(r"\bRETURN\b", query_clean, re.IGNORECASE))
        if not return_matches:
            logger.debug("Ricostruzione: nessun RETURN trovato")
            return {}

        last_return_pos = return_matches[-1].start()
        prefix = query_clean[:last_return_pos].strip()

        # STEP 1: Assegna variabili ai nodi anonimi
        prefix_with_vars, var_counter = _assign_variables_to_anonymous_nodes(prefix)

        logger.debug(f"Query dopo assegnazione variabili:\n{prefix_with_vars}")

        # STEP 2: Assegna variabili alle relazioni anonime
        prefix_with_all_vars, _ = _assign_variables_to_anonymous_rels(prefix_with_vars)

        logger.debug(f"Query dopo assegnazione variabili:\n{prefix_with_all_vars}")

        # STEP 3: Estrai TUTTE le variabili (incluse quelle appena create)
        node_vars, rel_vars = _extract_variables_from_query(prefix_with_all_vars)
        all_vars = node_vars + rel_vars

        if not all_vars:
            logger.debug("Ricostruzione: nessuna variabile trovata")
            return {}

        logger.info(f"Variabili per grafo - Nodi: {node_vars}, Relazioni: {rel_vars}")

        # STEP 4: Costruisci query che restituisce TUTTO (nodi + relazioni)
        attempt_vars: List[str] = []
        for var in all_vars:
            if var not in attempt_vars:
                attempt_vars.append(var)

        graph_records: List[Dict[str, Any]] = []
        removed_vars: List[str] = []

        while attempt_vars:
            graph_query = f"{prefix_with_all_vars}\nRETURN DISTINCT {', '.join(attempt_vars)} LIMIT 10"

            logger.info(f"Query ricostruita:\n{graph_query}")
            try:
                _ensure_read_only_query(graph_query)
                graph_records = execute_cypher_with_records(graph_query)
                break
            except Neo4jError as neo_exc:
                missing_var = _extract_missing_variable_from_error(str(neo_exc))
                if missing_var and missing_var in attempt_vars:
                    logger.warning(
                        "Ricostruzione grafo: variabile '%s' non definita, la rimuovo dal RETURN.",
                        missing_var,
                    )
                    attempt_vars = [v for v in attempt_vars if v != missing_var]
                    removed_vars.append(missing_var)
                    continue
                logger.error("Errore ricostruzione grafo: %s", neo_exc)
                raise

        if removed_vars:
            logger.warning(
                "Ricostruzione grafo completata senza le variabili %s (fuori scope).",
                removed_vars,
            )

        if not attempt_vars:
            logger.warning("Ricostruzione grafo: nessuna variabile valida disponibile.")
            return {}

        # subito dopo aver ottenuto graph_records
        try:
            if graph_records:
                sample_vals = list(graph_records[0].values())
                logger.warning(
                    "Sample record types: %s", [type(v) for v in sample_vals]
                )
            else:
                logger.warning("Nessun record restituito dalla query ricostruita")
        except Exception as e:
            logger.warning("Impossibile introspezionare i graph_records: %s", e)

        # Estrai il grafo
        snapshot = extract_graph_snapshot(graph_records)

        if not snapshot.get("nodes"):
            logger.warning(
                "Query eseguita ma snapshot vuoto - probabilmente nessun dato"
            )

        return snapshot

    except Exception as exc:
        logger.error(f"Errore ricostruzione grafo: {exc}", exc_info=True)
        return {}


def _assign_variables_to_anonymous_nodes(query: str) -> Tuple[str, int]:
    """
    Assegna variabili uniche ai nodi anonimi tipo (:Label) o ().
    Ritorna: (query_modificata, numero_variabili_aggiunte)

    Esempio:
    Input:  (c:Cliente)-[:REL]->(:Documento)
    Output: (c:Cliente)-[:REL]->(n1:Documento)
    """
    var_counter = 0
    result = query

    # Pattern per nodi anonimi: () o (:Label) ma NON (var) o (var:Label)
    # Cerchiamo ( seguito da : o ) ma NON da una variabile
    anonymous_pattern = re.compile(
        r"\("  # Aperta parentesi
        r"(?!"  # Negative lookahead
        r"[a-zA-Z_][a-zA-Z0-9_]*"  # NON una variabile
        r"[\s:]"  # seguita da spazio o :
        r")"
        r"(:[a-zA-Z0-9_`|]+)?"  # Optional label(s)
        r"(?:\s*\{[^}]*\})?"  # Optional properties
        r"\)"  # Chiusa parentesi
    )

    def replace_anonymous(match):
        nonlocal var_counter
        var_counter += 1
        var_name = f"n{var_counter}"

        # Se c'è un label, lo manteniamo
        labels = match.group(1) or ""  # Cattura :Label se presente
        return f"({var_name}{labels})"

    result = anonymous_pattern.sub(replace_anonymous, result)

    if var_counter > 0:
        logger.info(f"Assegnate {var_counter} variabili a nodi anonimi")

    return result, var_counter


def _assign_variables_to_anonymous_rels(
    query: str, rel_var_counter: int = 0
) -> Tuple[str, int]:
    """
    Assegna variabili uniche alle relazioni anonime tipo -[:TYPE]-> o -[]->.
    """
    result = query

    # Pattern per relazioni anonime: -[]- o -[:TYPE]- o -[:TYPE*1..2]- ecc.
    # NON matcha -[r]- o -[r:TYPE]-
    anonymous_pattern = re.compile(
        r"-\["  # Inizio relazione
        r"(?![a-zA-Z_][a-zA-Z0-9_]*)"  # Negative lookahead: NON una variabile
        r"([:[a-zA-Z0-9_`|*.. ]*)?"  # Gruppo 1: Optional type/spec (es. :REL, :REL*1..2)
        r"(\s*\{[^}]*\})?"  # Gruppo 2: Optional properties
        r"\s*\]-"  # Chiusa relazione
    )

    matches_to_replace = []
    for match in anonymous_pattern.finditer(result):
        matches_to_replace.append(match)

    for match in reversed(matches_to_replace):
        rel_var_counter += 1
        var_name = f"r{rel_var_counter}"

        rel_type = match.group(1) or ""  # Es. ":HA_RICEVUTO"
        props = match.group(2) or ""

        replacement = f"-[{var_name}{rel_type}{props}]-"

        result = result[: match.start()] + replacement + result[match.end() :]

    if (rel_var_counter - (rel_var_counter - len(matches_to_replace))) > 0:
        logger.info(
            f"Assegnate {len(matches_to_replace)} variabili a relazioni anonime"
        )

    return result, rel_var_counter


def build_graph_payload_from_results(
    query: str, records: List[Dict[str, Any]]
) -> Tuple[Dict[str, List[Dict[str, Any]]], str]:
    """
    Costruisce il payload del grafo dai risultati Neo4j.
    Ritorna: (graph_payload, source_type)
    """
    # PRIMA: prova a estrarre direttamente dai records
    graph_payload = extract_graph_snapshot(records)
    if graph_payload and graph_payload.get("nodes"):
        logger.info(f"Grafo estratto dai records: {len(graph_payload['nodes'])} nodi")
        return graph_payload, "context_results"

    # SECONDA: prova a ricostruire dalla query
    logger.info("Nessun grafo nei records, tento ricostruzione dalla query")
    rebuilt_graph = _reconstruct_graph_from_query(query)
    if rebuilt_graph and rebuilt_graph.get("nodes"):
        logger.info(f"Grafo ricostruito: {len(rebuilt_graph['nodes'])} nodi")
        return rebuilt_graph, "reconstructed_from_query"

    logger.warning("Nessun grafo trovato né nei records né nella ricostruzione")
    return {}, "no_nodes_detected"


def _extract_variables_from_query(query_prefix: str) -> Tuple[List[str], List[str]]:
    """
    Estrae variabili di nodi e relazioni da una query Cypher.
    Returns: (node_variables, relationship_variables)
    """
    # Pattern per nodi: (varName), (varName:Label), (varName:`Label`), etc.
    node_pattern = re.compile(
        r"\(([a-zA-Z_][a-zA-Z0-9_]*)"  # Nome variabile
        r"(?:\s*:[^)]+)?"  # Optional label(s)
        r"(?:\s*\{[^}]*\})?"  # Optional properties
        r"\)"
    )

    # Pattern per relazioni: -[varName]-, -[varName:TYPE]-, etc.
    rel_pattern = re.compile(
        r"-\[([a-zA-Z_][a-zA-Z0-9_]*)"  # Nome variabile
        r"(?:\s*:[^\]]+)?"  # Optional type(s)
        r"(?:\s*\{[^}]*\})?"  # Optional properties
        r"\]-"
    )

    node_vars: Set[str] = set()
    rel_vars: Set[str] = set()

    # Parole chiave Cypher da escludere
    cypher_keywords = {
        "match",
        "where",
        "with",
        "return",
        "optional",
        "union",
        "order",
        "limit",
        "skip",
        "and",
        "or",
        "not",
        "in",
        "as",
        "distinct",
        "by",
        "asc",
        "desc",
        "case",
        "when",
        "then",
        "else",
        "end",
        "create",
        "merge",
        "delete",
        "set",
        "remove",
    }

    # Trova tutti i nodi
    for match in node_pattern.finditer(query_prefix):
        var_name = match.group(1)
        if var_name and var_name.lower() not in cypher_keywords:
            node_vars.add(var_name)

    # Trova tutte le relazioni
    for match in rel_pattern.finditer(query_prefix):
        var_name = match.group(1)
        if var_name and var_name.lower() not in cypher_keywords:
            rel_vars.add(var_name)

    logger.debug(f"Estratte {len(node_vars)} var nodo, {len(rel_vars)} var relazione")

    return sorted(list(node_vars)), sorted(list(rel_vars))


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
    examples_top_k: Optional[int] = None  # override top-k per RAG (opzionale)


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
    generated_query: Optional[str] = None
    context_completo: List[Dict[str, Any]] = []
    prompt_examples: List[Dict[str, Any]] = []
    semantic_candidates: List[Dict[str, Any]] = []
    examples_similarity: Optional[float] = None
    specialist_route: Optional[str] = None

    user_id = user_question.user_id or "default_user"
    user_input_text = user_question.question.strip()
    effective_question_text = user_input_text
    scope_reset_flag = False
    requested_examples_top_k = (
        user_question.examples_top_k
        if user_question.examples_top_k and user_question.examples_top_k > 0
        else None
    )
    session = get_or_create_session(user_id)

    logger.info(f"📥 [{user_id}] Domanda: '{user_input_text}'")

    try:
        ## == FASE 1: GESTIONE PRELIMINARE (SMALL TALK & PERICOLO)

        # --- MODIFICA CHIAVE 1: GESTIONE SMALL TALK ANTICIPATA ---
        # Intercettiamo "Grazie mille" ecc. SUBITO, prima di qualsiasi logica di memoria.
        if _is_small_talk(effective_question_text):
            logger.info("Small talk rilevato: rotta diretta a GENERAL_CONVERSATION.")
            start_conv_time = time.perf_counter()

            # Chiamiamo l'agente con la domanda ORIGINALE (es. "Grazie mille")
            # e il prompt YAML che hai caricato (general_conversation_prompt.yaml)
            conversation_reply = run_general_conversation_agent(effective_question_text)

            timing_details["small_talk"] = True
            timing_details["conversazione"] = time.perf_counter() - start_conv_time
            timing_details["totale"] = time.perf_counter() - start_total_time

            add_message_to_session(
                user_id,
                effective_question_text,
                conversation_reply,
                [],
                "",
                meta={"event": "general_conversation", "small_talk": True},
            )

            response_payload = {
                "domanda": effective_question_text,
                "risposta": conversation_reply,
                "specialist": "GENERAL_CONVERSATION",
                "query_generata": None,
                "context": [],
                "graph_data": {},
                "timing_details": timing_details,
                "examples_top_k": None,
                "examples_used": [],
                "repair_audit": [],
                "success": True,
            }

            # Salva in cache anche questa risposta di cortesia
            if ENABLE_CACHE:
                cache_key = _make_cache_key(
                    user_id, effective_question_text, effective_question_text
                )
                query_cache[cache_key] = {
                    "response": copy.deepcopy(response_payload),
                    "context": [],
                    "graph_data": {},
                    "query_generata": None,
                    "specialist": "GENERAL_CONVERSATION",
                    "examples_top_k": None,
                }
            return response_payload
        # --- FINE BLOCCO SMALL TALK ---

        if _contains_dangerous_intent(effective_question_text):
            session.pending_confirmation = None
            warning_message = (
                "Posso solo consultare i dati in lettura. "
                "Per sicurezza non posso creare, modificare o cancellare informazioni nel database."
            )
            add_message_to_session(
                user_id,
                effective_question_text,
                warning_message,
                [],
                "",
                meta={
                    "event": "dangerous_intent",
                    "stage": "pre_translation",
                    "scope_reset": scope_reset_flag,
                },
            )
            timing_details["dangerous_intent"] = True
            timing_details["generazione_query"] = 0.0
            timing_details["esecuzione_db"] = 0.0
            timing_details["sintesi_risposta"] = 0.0
            timing_details["totale"] = time.perf_counter() - start_total_time
            timing_details["scope_reset"] = scope_reset_flag
            return {
                "domanda": effective_question_text,
                "query_generata": None,
                "risposta": warning_message,
                "context": [],
                "graph_data": {},
                "timing_details": timing_details,
                "specialist": "SAFETY_GUARD",
                "examples_top_k": requested_examples_top_k,
                "examples_used": [],
                "repair_audit": [],
                "success": False,
                "dangerous_intent": True,
            }

        start_preprocessing_time = time.perf_counter()

        # 1a. Gestione Memoria: Il nuovo Contextualizer riscrive la domanda
        chat_history = get_conversation_context(user_id, effective_question_text)
        timing_details["memory_used"] = bool(chat_history)
        question_text = run_contextualizer_agent(effective_question_text, chat_history)
        if question_text != effective_question_text:
            logger.info(f"Domanda riscritta dal Contestualizzatore: '{question_text}'")

        # 1b. Correzione Fuzzy
        question_text, correzioni = correggi_nomi_fuzzy_neo4j(question_text)
        if correzioni:
            logger.info(f"Correzioni Fuzzy applicate: {', '.join(correzioni)}")

        # 1c. Traduzione (fondamentale per la coerenza dei prompt)
        english_task = run_translator_agent(question_text)
        logger.info(f"Task tradotto in Inglese: '{english_task}'")

        if _contains_dangerous_intent(english_task):
            session.pending_confirmation = None
            warning_message = (
                "Posso aiutarti solo con analisi e ricerche. "
                "Non ho il permesso di creare, modificare o cancellare dati."
            )
            add_message_to_session(
                user_id,
                effective_question_text,
                warning_message,
                [],
                "",
                meta={
                    "event": "dangerous_intent",
                    "stage": "post_translation",
                    "scope_reset": scope_reset_flag,
                },
            )
            timing_details["dangerous_intent"] = True
            timing_details["generazione_query"] = 0.0
            timing_details["esecuzione_db"] = 0.0
            timing_details["sintesi_risposta"] = 0.0
            timing_details["totale"] = time.perf_counter() - start_total_time
            timing_details["scope_reset"] = scope_reset_flag
            return {
                "domanda": effective_question_text,
                "query_generata": None,
                "risposta": warning_message,
                "context": [],
                "graph_data": {},
                "timing_details": timing_details,
                "specialist": "SAFETY_GUARD",
                "examples_top_k": requested_examples_top_k,
                "examples_used": [],
                "repair_audit": [],
                "success": False,
                "dangerous_intent": True,
            }

        effective_examples_top_k = (
            requested_examples_top_k
            if requested_examples_top_k is not None
            else example_retriever.get_default_top_k()
        )

        cache_key = None
        if ENABLE_CACHE:
            cache_scope_user = user_id if ENABLE_MEMORY else "single_turn"
            scope_token = (
                f"{cache_scope_user}|k{effective_examples_top_k}"
                if effective_examples_top_k is not None
                else cache_scope_user
            )
            cache_key = _make_cache_key(scope_token, question_text, english_task)
        cached_entry = None
        if ENABLE_CACHE and cache_key and cache_key in query_cache:
            cached_entry = copy.deepcopy(query_cache[cache_key])
            if cached_entry:
                logger.info(
                    f"♻️ Cache HIT per utente {user_id}. Riutilizzo risposta pre-calcolata."
                )
                cached_response = cached_entry.get("response") or {}
                cached_timings = cached_response.get("timing_details", {})
                timings_copy = dict(cached_timings)
                timings_copy["cache_hit"] = True
                cached_response["timing_details"] = timings_copy
                cached_context = cached_response.get("context")
                if cached_context is None:
                    cached_context = cached_entry.get("context", [])
                cached_graph = cached_response.get("graph_data")
                if cached_graph is None:
                    cached_graph = cached_entry.get("graph_data", {})
                cached_examples = cached_response.get("examples_used")
                if cached_examples is None:
                    cached_examples = cached_entry.get("examples_used", [])

                add_message_to_session(
                    user_id,
                    effective_question_text,
                    cached_response.get("risposta", ""),
                    cached_context,
                    cached_entry.get("query_generata", ""),
                    meta={
                        "event": "cache_hit",
                        "cache_key": cache_key,
                        "scope_reset": scope_reset_flag,
                    },
                )

                return {
                    "domanda": effective_question_text,
                    "query_generata": cached_entry.get("query_generata"),
                    "risposta": cached_response.get("risposta"),
                    "context": cached_context,
                    "graph_data": cached_graph,
                    "timing_details": timings_copy,
                    "specialist": cached_entry.get("specialist"),
                    "examples_top_k": cached_entry.get("examples_top_k"),
                    "examples_used": cached_examples,
                    "cached": True,
                }

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
        """
        generated_query = run_coder_agent(
            question=english_task,
            relevant_schema=relevant_schema,
            prompt_template=coder_prompt_template,
        )
        """
        (
            generated_query,
            prompt_examples,
            semantic_candidates,
            examples_similarity,
        ) = run_coder_agent(
            question=english_task,
            relevant_schema=relevant_schema,
            prompt_template=coder_prompt_template,
            specialist_type=specialist_route,
            original_question_it=effective_question_text,
            examples_top_k=effective_examples_top_k,
        )

        timing_details["generazione_query"] = time.perf_counter() - start_gen_time
        if examples_similarity is not None:
            timing_details["examples_similarity"] = examples_similarity
        logger.info(
            f"Query generata dallo specialista '{specialist_route}':\n{generated_query}"
        )

        complexity_level, complexity_reasons = _classify_query_complexity(
            generated_query
        )
        timing_details["query_complexity"] = {
            "level": complexity_level,
            "reasons": complexity_reasons,
        }
        default_timeout = get_query_timeout_seconds()
        if not default_timeout or default_timeout <= 0:
            default_timeout = 30.0
        timeout_budget = float(default_timeout)
        if complexity_level == "critical":
            timeout_budget = min(timeout_budget, QUERY_CRITICAL_TIMEOUT)
        elif complexity_level == "risky":
            timeout_budget = min(timeout_budget, QUERY_RISKY_TIMEOUT)
        timing_details["query_timeout_seconds"] = timeout_budget
        timing_details["scope_reset"] = scope_reset_flag

        query_meta: Dict[str, Any] = {
            "complexity": {
                "level": complexity_level,
                "reasons": complexity_reasons,
                "timeout_seconds": timeout_budget,
            },
            "scope_reset": scope_reset_flag,
        }

        blocked_keyword = _detect_write_operation(generated_query)
        if blocked_keyword:
            logger.warning(
                "Query bloccata per operazione proibita '%s'.", blocked_keyword
            )
            warning_message = (
                "Non posso eseguire questa richiesta perché contiene operazioni "
                "che modificherebbero il database (es. "
                f"{blocked_keyword.upper()}). Posso solo consultare i dati in lettura."
            )
            timing_details["esecuzione_db"] = 0.0
            timing_details["sintesi_risposta"] = 0.0
            timing_details["totale"] = time.perf_counter() - start_total_time

            add_message_to_session(
                user_id,
                effective_question_text,
                warning_message,
                [],
                generated_query,
                meta={**query_meta, "blocked_keyword": blocked_keyword},
            )

            return {
                "domanda": effective_question_text,
                "query_generata": generated_query,
                "risposta": warning_message,
                "context": [],
                "graph_data": {},
                "timing_details": timing_details,
                "specialist": specialist_route,
                "examples_top_k": effective_examples_top_k,
                "examples_used": [
                    {
                        "id": ex.get("id"),
                        "question": ex.get("question"),
                        "similarity": ex.get("similarity"),
                    }
                    for ex in (prompt_examples or [])
                ],
                "repair_audit": [],
                "success": False,
                "dangerous_intent": True,
            }

        ## == FASE 3: ESECUZIONE E SINTESI DELLA RISPOSTA
        # 3a. Esecuzione query con ciclo di repair (configurabile)
        start_db_time = time.perf_counter()
        retry_cfg = config.system.get("retry", {}) or {}
        semantic_cfg = retry_cfg.get("semantic_expansion", {}) or {}
        max_recent = int(retry_cfg.get("max_recent_repairs", 0))
        hints_texts = []
        for hint_file in retry_cfg.get("hints_files", []) or []:
            try:
                with open(hint_file, "r", encoding="utf-8") as hf:
                    hints_texts.append(hf.read())
            except Exception:
                continue
        recent_repairs = _load_recent_repairs(max_recent) if max_recent > 0 else ""
        base_hints_text = "\n\n".join(hints_texts).strip()

        execution_attempts: List[Dict[str, Any]] = []
        context_completo: List[Dict[str, Any]] = []
        repair_audit: List[Dict[str, str]] = []
        safe_rewrite_applied = False
        safe_rewrite_notes: List[str] = []
        slow_query_message: Optional[str] = None

        try:
            final_query, context_completo, repair_audit = (
                execute_query_with_iterative_improvement(
                    initial_query=generated_query,
                    english_task=english_task,
                    original_question_it=effective_question_text,
                    relevant_schema=relevant_schema,
                    retry_cfg=retry_cfg,
                    base_hints_text=base_hints_text,
                    recent_repairs=recent_repairs,
                    semantic_cfg=semantic_cfg,
                    semantic_examples=semantic_candidates,
                    contextualized_question=question_text,
                    execution_timeout=timeout_budget,
                )
            )
            execution_attempts.append({"query": generated_query, "status": "success"})
            generated_query = final_query
        except QueryTimeoutError as timeout_exc:
            timing_details["initial_timeout"] = str(timeout_exc)
            execution_attempts.append(
                {
                    "query": generated_query,
                    "status": "timeout",
                    "message": str(timeout_exc),
                }
            )
            rewritten_query, rewrite_notes = _rewrite_query_safe(
                generated_query, complexity_reasons
            )
            if rewritten_query:
                safe_rewrite_applied = True
                safe_rewrite_notes = rewrite_notes
                timing_details["safe_rewrite"] = {
                    "applied": True,
                    "notes": rewrite_notes,
                }
                try:
                    final_query, context_completo, repair_audit = (
                        execute_query_with_iterative_improvement(
                            initial_query=rewritten_query,
                            english_task=english_task,
                            original_question_it=effective_question_text,
                            relevant_schema=relevant_schema,
                            retry_cfg=retry_cfg,
                            base_hints_text=base_hints_text,
                            recent_repairs=recent_repairs,
                            semantic_cfg=semantic_cfg,
                            semantic_examples=semantic_candidates,
                            contextualized_question=question_text,
                            execution_timeout=timeout_budget,
                        )
                    )
                    execution_attempts.append(
                        {"query": rewritten_query, "status": "success"}
                    )
                    generated_query = final_query
                except QueryTimeoutError as timeout_exc_2:
                    execution_attempts.append(
                        {
                            "query": rewritten_query,
                            "status": "timeout",
                            "message": str(timeout_exc_2),
                        }
                    )
                    slow_query_message = (
                        "La query è stata interrotta perché troppo pesante. "
                        "Aggiungi filtri come periodo, fornitore o restringi la famiglia di prodotti prima di riprovare."
                    )
            else:
                slow_query_message = (
                    "La query è stata interrotta perché troppo pesante. "
                    "Aggiungi filtri come periodo, fornitore o restringi il perimetro prima di riprovare."
                )
        except UnsafeCypherError as unsafe_exc:
            execution_attempts.append(
                {
                    "query": generated_query,
                    "status": "unsafe",
                    "keyword": unsafe_exc.keyword,
                }
            )
            logger.warning("Blocco esecuzione query: %s", unsafe_exc)
            timing_details["esecuzione_db"] = time.perf_counter() - start_db_time
            timing_details["sintesi_risposta"] = 0.0
            timing_details["totale"] = time.perf_counter() - start_total_time
            warning_message = (
                "Per motivi di sicurezza non posso eseguire query che modificano "
                "il database. Ho rilevato l'operazione vietata "
                f"«{unsafe_exc.keyword.upper()}»."
            )
            add_message_to_session(
                user_id,
                effective_question_text,
                warning_message,
                [],
                generated_query,
                meta={**query_meta, "blocked_keyword": unsafe_exc.keyword},
            )
            return {
                "domanda": effective_question_text,
                "query_generata": generated_query,
                "risposta": warning_message,
                "context": [],
                "graph_data": {},
                "timing_details": timing_details,
                "specialist": specialist_route,
                "examples_top_k": effective_examples_top_k,
                "examples_used": [
                    {
                        "id": ex.get("id"),
                        "question": ex.get("question"),
                        "similarity": ex.get("similarity"),
                    }
                    for ex in (prompt_examples or [])
                ],
                "repair_audit": [],
                "success": False,
                "dangerous_intent": True,
            }

        if "safe_rewrite" not in timing_details:
            timing_details["safe_rewrite"] = {"applied": False}
        timing_details["execution_attempts"] = execution_attempts

        if safe_rewrite_applied:
            query_meta["safe_rewrite"] = {
                "applied": True,
                "notes": safe_rewrite_notes,
            }
        else:
            query_meta.setdefault("safe_rewrite", {"applied": False})

        if slow_query_message:
            elapsed_db = time.perf_counter() - start_db_time
            timing_details["esecuzione_db"] = elapsed_db
            timing_details["sintesi_risposta"] = 0.0
            timing_details["totale"] = time.perf_counter() - start_total_time
            timing_details["slow_query"] = {
                "message": slow_query_message,
                "attempts": execution_attempts,
                "complexity_level": complexity_level,
                "reasons": complexity_reasons,
            }
            _log_slow_query(
                {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "question": effective_question_text,
                    "english_task": english_task,
                    "query_originale": generated_query,
                    "attempts": execution_attempts,
                    "complexity": {
                        "level": complexity_level,
                        "reasons": complexity_reasons,
                    },
                    "scope_reset": scope_reset_flag,
                }
            )
            add_message_to_session(
                user_id,
                effective_question_text,
                slow_query_message,
                [],
                generated_query,
                meta={**query_meta, "slow_query": True, "attempts": execution_attempts},
            )
            return {
                "domanda": effective_question_text,
                "query_generata": generated_query,
                "risposta": slow_query_message,
                "context": [],
                "graph_data": {},
                "timing_details": timing_details,
                "specialist": specialist_route,
                "examples_top_k": effective_examples_top_k,
                "examples_used": [
                    {
                        "id": ex.get("id"),
                        "question": ex.get("question"),
                        "similarity": ex.get("similarity"),
                    }
                    for ex in (prompt_examples or [])
                ],
                "repair_audit": [],
                "success": False,
                "slow_query": True,
            }

        total_results = len(context_completo)
        timing_details["esecuzione_db"] = time.perf_counter() - start_db_time
        timing_details["repair_iterations"] = len(repair_audit)
        semantic_iterations = sum(
            1 for entry in repair_audit if entry.get("semantic_expansion")
        )
        if semantic_iterations:
            timing_details["semantic_expansion"] = True
            timing_details["semantic_expansion_iterations"] = semantic_iterations
        if retry_cfg.get("enabled", False) and repair_audit:
            logger.info(
                f"Iterative repair completato con {len(repair_audit)} tentativi intermedi."
            )
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

        graph_payload, graph_origin = build_graph_payload_from_results(
            generated_query, context_da_inviare
        )
        timing_details["graph_origin"] = graph_origin
        sanitized_context = make_context_json_serializable(context_da_inviare)
        context_str_for_llm = json.dumps(
            {"risultato": sanitized_context}, indent=2, ensure_ascii=False
        )

        start_synth_time = time.perf_counter()
        filters_summary = _scope_summary_for_synth(
            effective_question_text, question_text
        )

        final_answer = run_synthesizer_agent(
            question=effective_question_text,
            context_str=context_str_for_llm,
            total_results=total_results,
            contextualized_question=question_text,
            filters_summary=filters_summary,
        )
        timing_details["sintesi_risposta"] = time.perf_counter() - start_synth_time

        ## == FASE 4: FINALIZZAZIONE E SALVATAGGIO DELLA RISPOSTA
        timing_details["totale"] = time.perf_counter() - start_total_time
        logger.info(
            f"Risposta Finale generata in {timing_details['totale']:.2f}s: {final_answer}"
        )

        session_meta = {
            **query_meta,
            "execution_attempts": execution_attempts,
            "graph_origin": graph_origin,
            "specialist": specialist_route,
        }

        add_message_to_session(
            user_id,
            effective_question_text,
            final_answer,
            context_completo,
            generated_query,
            meta=session_meta,
        )

        examples_used_summary = [
            {
                "id": ex.get("id"),
                "question": ex.get("question"),
                "similarity": ex.get("similarity"),
            }
            for ex in (prompt_examples or [])
        ]

        response_payload = {
            "domanda": effective_question_text,
            "query_generata": generated_query,
            "risposta": final_answer,
            "context": sanitized_context,
            "graph_data": graph_payload,
            "timing_details": timing_details,
            "specialist": specialist_route,
            "examples_top_k": effective_examples_top_k,
            "examples_used": examples_used_summary,
            "repair_audit": repair_audit,
            "success": True,
        }

        if ENABLE_CACHE and cache_key:
            query_cache[cache_key] = {
                "response": copy.deepcopy(response_payload),
                "context": sanitized_context,
                "graph_data": graph_payload,
                "query_generata": generated_query,
                "specialist": specialist_route,
                "examples_top_k": response_payload["examples_top_k"],
                "examples_used": examples_used_summary,
            }
            logger.info("♻️ Cache STORE per query specialist")

        return response_payload

    except QueryTimeoutError as e:
        logger.error(
            f"Timeout durante l'esecuzione della query Neo4j: {e}", exc_info=False
        )
        timing_details.setdefault("errore", str(e))
        timing_details.setdefault("totale", time.perf_counter() - start_total_time)
        polite_msg = "Sto impiegando troppo tempo per recuperare i dati. Prova a restringere i filtri o a porre la domanda in modo più specifico."
        return {
            "domanda": effective_question_text,
            "query_generata": generated_query,
            "risposta": polite_msg,
            "context": [],
            "graph_data": {},
            "timing_details": timing_details,
            "specialist": specialist_route,
            "examples_top_k": requested_examples_top_k,
            "examples_used": [
                {
                    "id": ex.get("id"),
                    "question": ex.get("question"),
                    "similarity": ex.get("similarity"),
                }
                for ex in (prompt_examples or [])
            ],
            "repair_audit": [],
            "success": False,
        }
    except Exception as e:
        logger.error(
            f"ERRORE GRAVE nel flusso ad agenti specializzati: {e}", exc_info=True
        )
        timing_details.setdefault("errore", str(e))
        timing_details.setdefault("totale", time.perf_counter() - start_total_time)
        polite_msg = "Al momento non riesco a rispondere a questa domanda. Riprova più tardi o riformula la richiesta."
        return {
            "domanda": effective_question_text,
            "query_generata": generated_query,
            "risposta": polite_msg,
            "context": [],
            "graph_data": {},
            "timing_details": timing_details,
            "specialist": specialist_route,
            "examples_top_k": requested_examples_top_k,
            "examples_used": [
                {
                    "id": ex.get("id"),
                    "question": ex.get("question"),
                    "similarity": ex.get("similarity"),
                }
                for ex in (prompt_examples or [])
            ],
            "repair_audit": [],
            "success": False,
        }


@app.post("/feedback")
async def submit_feedback(payload: FeedbackPayload):
    """Raccoglie il feedback esplicito degli utenti."""

    if not FEEDBACK_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="La raccolta feedback è temporaneamente disabilitata.",
        )

    category = (payload.category or "").strip().lower()
    if category not in FEEDBACK_ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Categoria feedback non supportata: {payload.category}",
        )

    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": payload.user_id,
        "question": payload.question,
        "category": category,
        "notes": payload.notes,
        "answer": payload.answer,
        "query_generated": payload.query_generated,
        "metadata": payload.metadata,
    }
    _store_feedback_entry(entry)
    logger.info(
        "Feedback registrato per utente %s (categoria=%s)", payload.user_id, category
    )
    return {"status": "ok", "stored": True}


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
        "top_k": 5  # opzionale, default configurato (EXAMPLES_TOP_K o 3)
    }
    """
    question = request.get("question")
    specialist = request.get("specialist")
    top_k = request.get("top_k")
    if top_k is None:
        top_k = example_retriever.get_default_top_k()

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    # Se specialist non specificato, usa router
    if not specialist:
        english_task = run_translator_agent(question)
        specialist = run_specialist_router_agent(english_task)

    # Recupera esempi
    examples = example_retriever.retrieve(
        question,
        specialist,
        top_k=top_k,
        allow_low_similarity=True,
    )
    best_similarity = example_retriever.last_similarity

    return {
        "question": question,
        "specialist_used": specialist,
        "top_k": top_k,
        "best_similarity": best_similarity,
        "min_similarity": getattr(example_retriever, "min_similarity", None),
        "retrieved_examples": [
            {
                "rank": i + 1,
                "id": ex["id"],
                "question": ex["question"],
                "cypher": ex["cypher"],
                "similarity": ex.get("similarity"),
            }
            for i, ex in enumerate(examples)
        ],
        "total_available": len(
            example_retriever.examples_by_specialist.get(specialist, [])
        ),
    }


# =======================================================================
