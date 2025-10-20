"""
Script di validazione universale per confrontare approcci Specialist vs Hybrid.

    # Testa una singola config specialist(es. specialist-gpt4o-mini):
    python run_validation.py --approach specialist --config gpt4o-mini

    Testa una singola config hybrid (es. hybrid-gpt4o-mini):
    python run_validation.py --approach hybrid --config gpt4o-mini

    # Confronta specialist vs hybrid con stessa config
    python run_validation.py --compare-approaches

    # Testa tutte le config specialist
    python run_validation.py --approach specialist --compare-all

    # Testa tutte le config hybrid
    python run_validation.py --approach hybrid --compare-all

    # Confronta 2 config specifiche
    python run_validation.py --compare specialist-gpt4o-mini hybrid-gpt4o-mini

"""

import json
import math
import requests
from neo4j import GraphDatabase
import argparse
from datetime import datetime
from pathlib import Path
import yaml
import time
from typing import List, Dict, Any, Tuple

# Neo4j temporal types for JSON serialization safety
try:
    from neo4j.time import (
        Date as Neo4jDate,
        DateTime as Neo4jDateTime,
        Time as Neo4jTime,
        Duration as Neo4jDuration,
    )
except Exception:  # pragma: no cover - fallback if import shape changes
    Neo4jDate = Neo4jDateTime = Neo4jTime = Neo4jDuration = tuple()


def _json_default(o):
    """Fallback serializer for JSON: handles Neo4j temporal types and sets.

    - Neo4j temporal objects and Python datetime are converted to ISO strings
    - sets are converted to sorted lists
    - unknown objects are stringified as a safe fallback
    """
    try:
        # Python datetime
        if isinstance(o, datetime):
            return o.isoformat()
        # Neo4j temporal types
        if isinstance(o, (Neo4jDate, Neo4jDateTime, Neo4jTime, Neo4jDuration)):
            return str(o)
        # sets -> list
        if isinstance(o, set):
            return sorted(list(o))
    except Exception:
        pass
    # Generic fallback
    return str(o)


from difflib import SequenceMatcher

# =======================================================================
# CONFIGURAZIONE
# =======================================================================
# Path di questo script (validation/)
SCRIPT_DIR = Path(__file__).parent

# Path delle cartelle dei chatbot
SPECIALIST_DIR = SCRIPT_DIR.parent / "chatbot_specialist"
HYBRID_DIR = SCRIPT_DIR.parent / "chatbot_hybrid"

# Output directories (dentro validation/)
# VALIDATION_SET = SCRIPT_DIR / "validation_set_2.json"
VALIDATION_SET = SCRIPT_DIR / "validation_set.json"
RESULTS_DIR = SCRIPT_DIR / "validation_results"
RESULTS_DIR.mkdir(exist_ok=True)

# API URL (uguale per entrambi)
API_URL = "http://127.0.0.1:8000"

# Neo4j
try:
    from dotenv import load_dotenv
    import os

    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Blue2018")
except:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Blue2018"

# =======================================================================
# CONFIGURAZIONI MODELLI
# =======================================================================

# Config per approccio SPECIALIST (con router)
CONFIGS_SPECIALIST = {
    "gpt4o-mini": {
        "contextualizer": "gpt-4o-mini",
        "router": "gpt-4o-mini",
        "coder": "gpt-4o-mini",
        "general_conversation": "gpt-4o-mini",
        "synthesizer": "gpt-4o-mini",
        "translator": "gpt-4o-mini",
    },
    "gemini-2.5-pro": {
        "contextualizer": "gemini-2.5-pro",
        "router": "gemini-2.5-pro",
        "coder": "gemini-2.5-pro",
        "general_conversation": "gemini-2.5-pro",
        "synthesizer": "gemini-2.5-pro",
        "translator": "gemini-2.5-pro",
    },
    "gemini-2.5-flash": {
        "contextualizer": "gemini-2.5-flash",
        "router": "gemini-2.5-flash",
        "coder": "gemini-2.5-flash",
        "general_conversation": "gemini-2.5-flash",
        "synthesizer": "gemini-2.5-flash",
        "translator": "gemini-2.5-pro",
    },
    "gpt4o": {
        "contextualizer": "gpt-4o",
        "router": "gpt-4o",
        "coder": "gpt-4o",
        "general_conversation": "gpt-4o",
        "synthesizer": "gpt-4o",
        "translator": "gpt-4o",
    },
    "llama3-coder": {
        "contextualizer": "gpt-4o-mini",
        "router": "llama3",
        "coder": "llama3",
        "general_conversation": "llama3",
        "synthesizer": "llama3",
        "translator": "llama3",
    },
    "llama3-8b-vertex": {
        "contextualizer": "llama3-8b-vertex",
        "coder": "llama3-8b-vertex",
        "router": "gpt-4o-mini",
        "general_conversation": "gemini-1.5-pro",
        "synthesizer": "gpt-4o-mini",
        "translator": "llama3-8b-vertex",
    },
}

# Config per approccio HYBRID (senza router)
CONFIGS_HYBRID = {
    "gpt4o-mini": {
        "contextualizer": "gpt-4o-mini",
        "coder": "gpt-4o-mini",
        "synthesizer": "llama3",
        "translator": "gpt-4o-mini",
    },
    "gpt4o-coder": {
        "contextualizer": "gpt-4o-mini",
        "coder": "gpt-4o",
        "synthesizer": "llama3",
        "translator": "gpt-4o-mini",
    },
    "llama3-coder": {
        "contextualizer": "gpt-4o-mini",
        "coder": "llama3",
        "synthesizer": "gpt-4o-mini",
        "translator": "gpt-4o-mini",
    },
    "llama3-full": {
        "contextualizer": "llama3",
        "coder": "llama3",
        "synthesizer": "llama3",
        "translator": "llama3",
    },
    "gpt4o-full": {
        "contextualizer": "gpt-4o",
        "coder": "gpt-4o",
        "synthesizer": "gpt-4o",
        "translator": "gpt-4o",
    },
    "llama3-8b-vertex": {
        "contextualizer": "llama3-8b-vertex",
        "coder": "llama3-8b-vertex",
        "synthesizer": "gpt-4o-mini",
        "translator": "llama3-8b-vertex",
    },
}

# =======================================================================
# UTILITY FUNCTIONS
# =======================================================================


def compare_results(
    result1: List[Dict[str, Any]], result2: List[Dict[str, Any]]
) -> bool:
    """Confronta due risultati ignorando alias e ordine."""
    if len(result1) != len(result2):
        return False
    if not result1:
        return True

    try:
        if len(result1[0]) != len(result2[0]):
            return False
        # Normalizza gli scalari (es. "10" -> 10.0) e confronta come insiemi senza ordine
        values1 = {_canonical_row_values(d) for d in result1}
        values2 = {_canonical_row_values(d) for d in result2}
        return values1 == values2
    except Exception:
        # Fallback robusto: ordina per chiave e normalizza i valori
        set1 = {_canonical_row_items(d) for d in result1}
        set2 = {_canonical_row_items(d) for d in result2}
        return set1 == set2


def explain_mismatch(
    result1: List[Dict[str, Any]], result2: List[Dict[str, Any]], max_rows: int = 3
) -> Dict[str, Any]:
    """Restituisce un riassunto delle differenze: mancanti vs extra (sample)."""
    summary: Dict[str, Any] = {"missing": [], "extra": [], "note": None}
    if not result1 and not result2:
        return summary

    try:
        vals1 = {tuple(d.values()): d for d in result1}
        vals2 = {tuple(d.values()): d for d in result2}
        missing = [vals2[k] for k in vals2.keys() - vals1.keys()]
        extra = [vals1[k] for k in vals1.keys() - vals2.keys()]
    except Exception:
        set1 = {tuple(sorted(d.items())): d for d in result1}
        set2 = {tuple(sorted(d.items())): d for d in result2}
        missing = [set2[k] for k in set2.keys() - set1.keys()]
        extra = [set1[k] for k in set1.keys() - set2.keys()]

    summary["missing"] = missing[:max_rows]
    summary["extra"] = extra[:max_rows]
    if missing or extra:
        summary["note"] = (
            f"missing={len(missing)}, extra={len(extra)} (showing up to {max_rows})"
        )
    return summary


def _normalize_rows(result: List[Dict[str, Any]]) -> set:
    """Converte righe query in tuple ordinabili per comparazione insiemi."""
    normalized = set()
    for row in result:
        try:
            items = tuple(
                sorted(((str(k), _normalize_scalar(v)) for k, v in row.items()))
            )
            normalized.add(items)
        except Exception:
            normalized.add(tuple(sorted((str(k), str(v)) for k, v in row.items())))
    return normalized


def jaccard_similarity(
    result1: List[Dict[str, Any]], result2: List[Dict[str, Any]]
) -> float:
    """Calcola indice di Jaccard tra due liste di risultati."""
    set1 = _normalize_rows(result1)
    set2 = _normalize_rows(result2)
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    if not union:
        return 1.0
    intersection = set1 & set2
    return len(intersection) / len(union)


def jaccard_value_only(
    result1: List[Dict[str, Any]], result2: List[Dict[str, Any]]
) -> float:
    """Indice di Jaccard ignorando gli alias delle colonne (value-only).

    Confronta i risultati convertendo ogni riga in una tupla canonica di soli
    valori normalizzati e ordinati stabilmente. L'ordine delle righe è ignorato.
    """
    set1 = {_canonical_row_values(d) for d in (result1 or [])}
    set2 = {_canonical_row_values(d) for d in (result2 or [])}
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    if not union:
        return 1.0
    intersection = set1 & set2
    return len(intersection) / len(union)


def jaccard_on_expected_columns(
    gen_res: List[Dict[str, Any]],
    exp_res: List[Dict[str, Any]],
) -> float:
    """Indice di Jaccard proiettando la generata sulle colonne attese.

    - Usa come schema le chiavi presenti nella prima riga dell'atteso.
    - Proietta le righe della generata solo se includono tutte le colonne attese.
    - Confronta gli insiemi di tuple (k, valore_normalizzato) ordinate per chiave.
    - Ignora l'ordine delle righe.
    """
    expected_keys: List[str] = []
    if exp_res and isinstance(exp_res[0], dict):
        expected_keys = list(exp_res[0].keys())

    def _project_row(row: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        # Richiede la presenza di tutte le chiavi attese
        for k in expected_keys:
            if k not in row:
                return tuple()
        return tuple(
            sorted(((k, _normalize_scalar(row.get(k))) for k in expected_keys))
        )

    exp_set: set = set()
    gen_set: set = set()

    if exp_res:
        exp_set = {_project_row(r) for r in exp_res}
        # L'atteso deve sempre avere tutte le colonne: rimuovi eventuali sentinel vuoti
        exp_set.discard(tuple())
    if gen_res and expected_keys:
        gen_set = {_project_row(r) for r in gen_res}
        # Righe senza colonne attese generano sentinel vuoto: rimuovi
        gen_set.discard(tuple())

    if not exp_set and not gen_set:
        return 1.0
    union = exp_set | gen_set
    if not union:
        return 1.0
    intersection = exp_set & gen_set
    return len(intersection) / len(union)


def query_string_similarity(expected: str, generated: str) -> float:
    """Calcola similarità tra stringhe di query utilizzando SequenceMatcher."""
    if not expected or not generated:
        return 0.0
    # Normalizza whitespace per confronti più robusti.
    expected_norm = " ".join(expected.split())
    generated_norm = " ".join(generated.split())
    return SequenceMatcher(None, expected_norm, generated_norm).ratio()


# =========================
# Normalizzazione e fallback
# =========================


def _is_number_like(x: Any) -> bool:
    """True se x è numero o stringa numerica (interi/float, include '10', '10.0', '1e3')."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return True
    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        try:
            float(s)
            return True
        except Exception:
            return False
    return False


def _to_number(x: Any) -> float:
    """Converte in float se possibile (stringhe numeriche incluse)."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        return float(s)
    raise ValueError("Not a number-like")


def _normalize_scalar(v: Any) -> Any:
    """Normalizza uno scalare per il confronto:
    - stringhe numeriche -> float
    - int/float -> float
    - datetime/Neo4j temporali -> stringa
    - stringhe -> strip/lower
    """
    try:
        if _is_number_like(v):
            num = _to_number(v)
            # Normalize floats to fixed precision to absorb tiny FP drift
            try:
                return round(float(num), 6)
            except Exception:
                return float(num)
    except Exception:
        pass
    # Datetime-like: fallback a stringa
    try:
        from datetime import datetime, date, time

        if isinstance(v, (datetime, date, time)):
            return str(v)
    except Exception:
        pass
    if isinstance(v, str):
        # Case-insensitive string comparison: normalize to lowercase
        return v.strip().lower()
    return v


def _canonical_row_values(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Restituisce una tupla canonica basata SOLO sui valori normalizzati, ordinati stabilmente.
    Ignora i nomi delle colonne per tollerare alias diversi.
    """
    vals = [_normalize_scalar(v) for v in row.values()]
    # Ordina in modo stabile usando tipo e rappresentazione stringa
    vals_sorted = sorted(vals, key=lambda x: (str(type(x)), str(x)))
    return tuple(vals_sorted)


def _canonical_row_items(row: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Tupla canonica (chiave, valore_normalizzato) ordinata per chiave: fallback conservativo."""
    return tuple(sorted(((str(k), _normalize_scalar(v)) for k, v in row.items())))


def compare_numeric_scalar(
    result1: List[Dict[str, Any]],
    result2: List[Dict[str, Any]],
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> bool:
    """Fallback: se entrambi i risultati sono una singola riga con UN solo campo numerico,
    considera PASS se i numeri coincidono entro la tolleranza, ignorando alias.
    """

    def _extract_single_numeric(result: List[Dict[str, Any]]) -> Any:
        if not result or len(result) != 1:
            return None
        row = result[0]
        numerics = []
        for v in row.values():
            if _is_number_like(v):
                try:
                    numerics.append(_to_number(v))
                except Exception:
                    pass
        # Richiedi esattamente un campo numerico per evitare ambiguità
        if len(numerics) == 1:
            return numerics[0]
        return None

    v1 = _extract_single_numeric(result1)
    v2 = _extract_single_numeric(result2)
    if v1 is None or v2 is None:
        return False
    return math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol)


def compare_subset_on_expected_columns(
    gen_res: List[Dict[str, Any]],
    exp_res: List[Dict[str, Any]],
) -> bool:
    """Confronta consentendo colonne extra nella generata.

    Regole:
    - Stesso numero di righe richiesto.
    - Ogni riga generata deve contenere almeno tutte le colonne presenti nell'atteso.
    - Confronto per insieme di righe, limitato alle colonne attese, con normalizzazione dei valori.
    - Ignora l'ordine delle righe.
    """
    if len(gen_res) != len(exp_res):
        return False
    if not exp_res:
        # Nessuna riga attesa: lascia al confronto principale gestire questo caso
        return False
    # Colonne attese dalla prima riga (si assume schema uniforme)
    expected_keys = list(exp_res[0].keys())

    def _project_row(row: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        # Richiede che tutte le chiavi attese siano presenti
        for k in expected_keys:
            if k not in row:
                # Mancano colonne attese → non compatibile
                return tuple()
        return tuple(
            sorted(((k, _normalize_scalar(row.get(k))) for k in expected_keys))
        )

    exp_set = {_project_row(r) for r in exp_res}
    gen_set = {_project_row(r) for r in gen_res}

    # Se qualche riga generata mancava di colonne attese, tuple() sentinel sarà presente
    if tuple() in gen_set:
        return False
    return exp_set == gen_set


def run_query(tx, query, params=None):
    """Esegue query e ritorna risultati."""
    result = tx.run(query, params)
    return [r.data() for r in result]


def clear_system_cache():
    """Pulisce cache API."""
    try:
        response = requests.delete(f"{API_URL}/cache", timeout=5)
        if response.status_code == 200:
            print("Cache API pulita")
            return True
    except:
        pass
    return False


def get_config_path(approach: str) -> Path:
    """Ritorna il path di config/models.yaml per l'approccio."""
    if approach == "specialist":
        return SPECIALIST_DIR / "config" / "models.yaml"
    elif approach == "hybrid":
        return HYBRID_DIR / "config" / "models.yaml"
    else:
        raise ValueError(f"Approccio sconosciuto: {approach}")


def update_model_config(approach: str, config_name: str):
    """Aggiorna config/models.yaml dell'approccio specificato."""
    config_path = get_config_path(approach)

    if not config_path.exists():
        print(f"Config non trovato: {config_path}")
        return False

    # Scegli config giusta
    configs = CONFIGS_SPECIALIST if approach == "specialist" else CONFIGS_HYBRID

    if config_name not in configs:
        print(f"Config '{config_name}' non esiste per approccio '{approach}'")
        return False

    # Leggi e aggiorna
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["agent_models"] = configs[config_name]

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config aggiornata: {approach}/{config_name}")
    return True


def prompt_restart_api(approach: str):
    """Chiede all'utente di riavviare l'API giusta."""
    if approach == "specialist":
        script = "chatbot_specialist/chatbot_specialist.py"
    else:
        script = "chatbot_hybrid/chatbot_hybrid.py"

    print(f"\n RIAVVIA L'API!")
    print(
        f"   cd {SPECIALIST_DIR.parent.name}/{approach.replace('specialist', 'chatbot_specialist').replace('hybrid', 'chatbot_hybrid')}"
    )
    print(
        f"   uvicorn {'chatbot_specialist' if approach == 'specialist' else 'chatbot_hybrid'}:app --reload"
    )
    input("\n   Premi ENTER quando pronto...")


def wait_for_api():
    """Attende che l'API sia pronta."""
    for i in range(10):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("API pronta!")
                return True
        except:
            print(f"⏳ Attendo API... ({i+1}/10)")
            time.sleep(2)
    return False


# =======================================================================
# VALIDAZIONE CORE
# =======================================================================


def run_validation(approach: str, config_name: str):
    """Esegue validazione."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_config_name = f"{approach}-{config_name}"

    print("\n" + "=" * 70)
    print(f" VALIDAZIONE: {full_config_name}")
    print("=" * 70)

    # Pulizia cache
    clear_system_cache()
    time.sleep(1)

    # Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connesso a Neo4j")
    except Exception as e:
        print(f"Neo4j: {e}")
        return None

    # Carica test
    with open(VALIDATION_SET, "r") as f:
        validation_data = json.load(f)

    total = len(validation_data)
    success = 0
    results = []

    print(f"Test totali: {total}\n")

    # Esegui test
    for i, test in enumerate(validation_data, 1):
        test_id = test["id"]
        question = test["question"]
        expected_cypher = test["expected_cypher"]

        print(f"[{i}/{total}] {test_id}")

        result = {
            "config": full_config_name,
            "test_id": test_id,
            "question": question,
            "success": False,
            "error": None,
            "generated_cypher": None,
            "expected_cypher": expected_cypher,
            "time_total": None,
            "time_generation": None,
            "time_db": None,
            "time_synthesis": None,
            "jaccard_index": None,
            "query_similarity": None,
        }

        try:
            # Chiama API
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "user_id": f"val_{test_id}_{timestamp}"},
                timeout=180,
            )

            if response.status_code != 200:
                body = None
                try:
                    body = response.json()
                except Exception:
                    body = response.text[:400]
                raise ValueError(f"API error {response.status_code}: {body}")

            data = response.json()
            generated = data.get("query_generata")
            timing = data.get("timing_details", {})

            result["generated_cypher"] = generated
            result["time_total"] = timing.get("totale")
            result["time_generation"] = timing.get("generazione_query")
            result["time_db"] = timing.get("esecuzione_db")
            result["time_synthesis"] = timing.get("sintesi_risposta")

            if not generated:
                raise ValueError("Query mancante")

            # Confronta
            with driver.session() as session:
                try:
                    gen_res = session.execute_read(run_query, generated)
                except Exception as e:
                    raise ValueError(f"Syntax error: {str(e)[:80]}")

                exp_res = session.execute_read(run_query, expected_cypher)

                # Metriche di similarità sui risultati e sulle query
                jac = jaccard_similarity(gen_res, exp_res)
                jac_val = jaccard_value_only(gen_res, exp_res)
                jac_proj = jaccard_on_expected_columns(gen_res, exp_res)
                qsim = query_string_similarity(expected_cypher, generated)
                result["jaccard_index"] = jac
                result["jaccard_value_only"] = jac_val
                result["jaccard_expected_projection"] = jac_proj
                result["query_similarity"] = qsim

                same = compare_results(gen_res, exp_res)
                if not same:
                    # Fallback 1: consenti colonne extra nella generata se i valori delle colonne attese coincidono
                    same = compare_subset_on_expected_columns(gen_res, exp_res)
                if not same:
                    # Fallback 2: se entrambe sono scalari numerici uguali entro tolleranza, considera PASS
                    same = compare_numeric_scalar(gen_res, exp_res)
                if same:
                    result["success"] = True
                    success += 1
                    print("  ✅ PASS")
                else:
                    # Spiega mismatch
                    diff = explain_mismatch(gen_res, exp_res, max_rows=3)
                    result["success"] = False
                    result["error"] = "Results mismatch"
                    result["mismatch_missing"] = diff.get("missing")
                    result["mismatch_extra"] = diff.get("extra")
                    print("  ❌ MISMATCH")
                    if diff.get("note"):
                        print(f"     diff: {diff['note']}")
                    if diff.get("missing"):
                        print(f"     missing sample: {diff['missing']}")
                    if diff.get("extra"):
                        print(f"     extra sample: {diff['extra']}")
                    # Non interrompo; salvo risultato e continuo

        except Exception as e:
            result["error"] = str(e)[:200]
            print(f"  FAIL: {str(e)[:120]}")

        results.append(result)

    driver.close()

    # Metriche
    accuracy = (success / total * 100) if total > 0 else 0
    avg_time = sum(r["time_total"] for r in results if r["time_total"]) / total
    # Medie indici (solo dove calcolati)
    jac_values = [
        r["jaccard_index"]
        for r in results
        if isinstance(r.get("jaccard_index"), (int, float))
    ]
    jac_val_values = [
        r["jaccard_value_only"]
        for r in results
        if isinstance(r.get("jaccard_value_only"), (int, float))
    ]
    jac_proj_values = [
        r["jaccard_expected_projection"]
        for r in results
        if isinstance(r.get("jaccard_expected_projection"), (int, float))
    ]
    qsim_values = [
        r["query_similarity"]
        for r in results
        if isinstance(r.get("query_similarity"), (int, float))
    ]
    avg_jaccard = (sum(jac_values) / len(jac_values)) if jac_values else None
    avg_jaccard_value_only = (
        (sum(jac_val_values) / len(jac_val_values)) if jac_val_values else None
    )
    avg_jaccard_expected_projection = (
        (sum(jac_proj_values) / len(jac_proj_values)) if jac_proj_values else None
    )
    avg_qsim = (sum(qsim_values) / len(qsim_values)) if qsim_values else None

    # Salva
    output_file = RESULTS_DIR / f"{full_config_name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "config": full_config_name,
                "approach": approach,
                "config_name": config_name,
                "timestamp": timestamp,
                "summary": {
                    "total": total,
                    "success": success,
                    "failures": total - success,
                    "accuracy": accuracy,
                    "avg_time": avg_time,
                    "avg_jaccard": avg_jaccard,
                    "avg_jaccard_value_only": avg_jaccard_value_only,
                    "avg_jaccard_expected_projection": avg_jaccard_expected_projection,
                    "avg_query_similarity": avg_qsim,
                },
                "results": results,
            },
            f,
            indent=2,
            default=_json_default,
        )

    # Scrive anche un report testuale/Markdown affiancato al JSON, con le metriche chiave
    report_md = RESULTS_DIR / f"{full_config_name}_{timestamp}.md"
    try:
        lines = []
        lines.append(f"# Validation Report — {full_config_name}")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Total: {total}")
        lines.append(f"- Success: {success}")
        lines.append(f"- Failures: {total - success}")
        lines.append(f"- Accuracy: {accuracy:.1f}%")
        lines.append(f"- Avg Time: {avg_time:.2f}s")
        if avg_jaccard is not None:
            lines.append(f"- Avg Jaccard: {avg_jaccard:.3f}")
        if avg_jaccard_value_only is not None:
            lines.append(f"- Avg Jaccard (value-only): {avg_jaccard_value_only:.3f}")
        if avg_jaccard_expected_projection is not None:
            lines.append(
                f"- Avg Jaccard (proj exp cols): {avg_jaccard_expected_projection:.3f}"
            )
        if avg_qsim is not None:
            lines.append(f"- Avg Query Similarity: {avg_qsim:.3f}")
        lines.append("")
        lines.append("## Tests")
        lines.append(
            "| # | Test ID | PASS | Jac | JacVal | JacProj | QuerySim | Gen Time (s) | Note |"
        )
        lines.append(
            "|:-:|:--------|:----:|:---:|:------:|:-------:|:--------:|:------------:|:-----|"
        )
        for idx, r in enumerate(results, 1):
            ok = "✅" if r.get("success") else "❌"
            jac = r.get("jaccard_index")
            jac_val = r.get("jaccard_value_only")
            jac_proj = r.get("jaccard_expected_projection")
            qsim = r.get("query_similarity")
            jac_s = f"{jac:.3f}" if isinstance(jac, (int, float)) else "-"
            jac_val_s = f"{jac_val:.3f}" if isinstance(jac_val, (int, float)) else "-"
            jac_proj_s = (
                f"{jac_proj:.3f}" if isinstance(jac_proj, (int, float)) else "-"
            )
            qsim_s = f"{qsim:.3f}" if isinstance(qsim, (int, float)) else "-"
            tgen = r.get("time_total")
            tgen_s = f"{tgen:.2f}" if isinstance(tgen, (int, float)) else "-"
            note = r.get("error") or ""
            lines.append(
                f"| {idx} | {r.get('test_id','')} | {ok} | {jac_s} | {jac_val_s} | {jac_proj_s} | {qsim_s} | {tgen_s} | {note} |"
            )
        with open(report_md, "w", encoding="utf-8") as rf:
            rf.write("\n".join(lines) + "\n")
    except Exception:
        # Il report testo non deve bloccare la validazione
        pass

    print("\n" + "=" * 70)
    print("RISULTATI")
    print("=" * 70)
    print(f"Config:    {full_config_name}")
    print(f"Accuracy:  {accuracy:.1f}% ({success}/{total})")
    print(f"Avg Time:  {avg_time:.2f}s")
    if avg_jaccard is not None:
        print(f"Avg Jaccard: {avg_jaccard:.3f}")
    if avg_jaccard_value_only is not None:
        print(f"Avg Jaccard (value-only): {avg_jaccard_value_only:.3f}")
    if avg_jaccard_expected_projection is not None:
        print(f"Avg Jaccard (proj exp cols): {avg_jaccard_expected_projection:.3f}")
    if avg_qsim is not None:
        print(f"Avg QuerySim: {avg_qsim:.3f}")
    print(f"Saved:     {output_file.name}")

    return {
        "config": full_config_name,
        "approach": approach,
        "config_name": config_name,
        "accuracy": accuracy,
        "success": success,
        "total": total,
        "avg_time": avg_time,
    }


# =======================================================================
# MAIN
# =======================================================================


def main():
    parser = argparse.ArgumentParser(description="Validation universale")
    parser.add_argument(
        "--approach", choices=["specialist", "hybrid"], help="Approccio da testare"
    )
    parser.add_argument("--config", help="Config da testare")
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Confronta config specifiche (formato: specialist-gpt4o-mini hybrid-gpt4o-mini)",
    )
    parser.add_argument(
        "--compare-approaches",
        action="store_true",
        help="Confronta specialist-gpt4o-mini vs hybrid-gpt4o-mini",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Testa tutte le config dell'approccio scelto",
    )
    parser.add_argument("--list", action="store_true", help="Lista config")

    args = parser.parse_args()

    if args.list:
        print("\nSPECIALIST Configs (with router):")
        for name in CONFIGS_SPECIALIST.keys():
            print(f"  - specialist-{name}")
        print("\nHYBRID Configs (without router):")
        for name in CONFIGS_HYBRID.keys():
            print(f"  - hybrid-{name}")
        return

    if args.compare_approaches:
        # Confronta rappresentativi
        print("Confronto Specialist vs Hybrid (entrambi gpt4o-mini)\n")

        # Test specialist
        print("1Testing SPECIALIST...")
        update_model_config("specialist", "gpt4o-mini")
        prompt_restart_api("specialist")
        wait_for_api()
        res1 = run_validation("specialist", "gpt4o-mini")

        # Test hybrid
        print("\nTesting HYBRID...")
        update_model_config("hybrid", "gpt4o-mini")
        prompt_restart_api("hybrid")
        wait_for_api()
        res2 = run_validation("hybrid", "gpt4o-mini")

        # Confronto
        if res1 and res2:
            print("\n" + "=" * 70)
            print("CONFRONTO FINALE")
            print("=" * 70)
            print(
                f"Specialist: {res1['accuracy']:.1f}% ({res1['success']}/{res1['total']})"
            )
            print(
                f"Hybrid:     {res2['accuracy']:.1f}% ({res2['success']}/{res2['total']})"
            )
            diff = res1["accuracy"] - res2["accuracy"]
            winner = "Specialist" if diff > 0 else "Hybrid"
            print(f"\n Winner: {winner} (+{abs(diff):.1f}%)")
        return

    if args.compare:
        # Parse formato: specialist-gpt4o-mini hybrid-gpt4o-mini
        all_results = []
        for full_name in args.compare:
            parts = full_name.split("-", 1)
            if len(parts) != 2:
                print(f"Formato errato: {full_name} (usa: specialist-gpt4o-mini)")
                continue

            approach, config = parts
            update_model_config(approach, config)
            prompt_restart_api(approach)
            wait_for_api()
            res = run_validation(approach, config)
            if res:
                all_results.append(res)

        # Summary
        if len(all_results) > 1:
            print("\n" + "=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)
            for r in all_results:
                print(
                    f"{r['config']:<30} {r['accuracy']:>5.1f}% ({r['success']}/{r['total']})"
                )
        return

    if args.approach and args.config:
        if args.compare_all:
            # Tutte le config dell'approccio
            configs = (
                CONFIGS_SPECIALIST if args.approach == "specialist" else CONFIGS_HYBRID
            )
            for cfg_name in configs.keys():
                update_model_config(args.approach, cfg_name)
                prompt_restart_api(args.approach)
                wait_for_api()
                run_validation(args.approach, cfg_name)
        else:
            # Singola config
            update_model_config(args.approach, args.config)
            prompt_restart_api(args.approach)
            wait_for_api()
            run_validation(args.approach, args.config)
        return

    # Help
    parser.print_help()
    print("\nEsempi:")
    print("  python run_validation.py --approach specialist --config gpt4o-mini")
    print("  python run_validation.py --approach hybrid --config gpt4o-mini")
    print("  python run_validation.py --compare-approaches")
    print(
        "  python run_validation.py --compare specialist-gpt4o-mini hybrid-gpt4o-coder"
    )


if __name__ == "__main__":
    main()
