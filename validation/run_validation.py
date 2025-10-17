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
import requests
from neo4j import GraphDatabase
import argparse
from datetime import datetime
from pathlib import Path
import yaml
import time

# =======================================================================
# CONFIGURAZIONE
# =======================================================================
# Path di questo script (validation/)
SCRIPT_DIR = Path(__file__).parent

# Path delle cartelle dei chatbot
SPECIALIST_DIR = SCRIPT_DIR.parent / "chatbot_specialist"
HYBRID_DIR = SCRIPT_DIR.parent / "chatbot_hybrid"

# Output directories (dentro validation/)
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
        "synthesizer": "llama3",
        "translator": "gpt-4o-mini",
    },
    "gpt4o-coder": {
        "contextualizer": "gpt-4o-mini",
        "router": "gpt-4o-mini",
        "coder": "gpt-4o",
        "synthesizer": "llama3",
        "translator": "gpt-4o-mini",
    },
    "gpt4o-full": {
        "contextualizer": "gpt-4o",
        "router": "gpt-4o",
        "coder": "gpt-4o",
        "synthesizer": "gpt-4o",
        "translator": "gpt-4o",
    },
    "llama3-coder": {
        "contextualizer": "gpt-4o-mini",
        "router": "gpt-4o-mini",
        "coder": "llama3",
        "synthesizer": "llama3",
        "translator": "gpt-4o-mini",
    },
    "gpt4o-full": {
        "contextualizer": "gpt-4o",
        "coder": "gpt-4o",
        "router": "gpt-4o",
        "synthesizer": "gpt-4o",
        "translator": "gpt-4o",
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
}

# =======================================================================
# UTILITY FUNCTIONS
# =======================================================================


def compare_results(result1, result2) -> bool:
    """Confronta due risultati ignorando alias e ordine."""
    if len(result1) != len(result2):
        return False
    if not result1:
        return True

    try:
        if len(result1[0]) != len(result2[0]):
            return False
        values1 = {tuple(d.values()) for d in result1}
        values2 = {tuple(d.values()) for d in result2}
        print(values1)
        print(values2)
        return values1 == values2
    except:
        set1 = {tuple(sorted(d.items())) for d in result1}
        set2 = {tuple(sorted(d.items())) for d in result2}
        return set1 == set2


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
        }

        try:
            # Chiama API
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "user_id": f"val_{test_id}_{timestamp}"},
                timeout=60,
            )

            if response.status_code != 200:
                raise ValueError(f"API error {response.status_code}")

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

                if compare_results(gen_res, exp_res):
                    result["success"] = True
                    success += 1
                    print("  ✅ PASS")
                else:
                    raise ValueError("Results mismatch")

        except Exception as e:
            result["error"] = str(e)[:200]
            print(f"  FAIL: {str(e)[:60]}")

        results.append(result)

    driver.close()

    # Metriche
    accuracy = (success / total * 100) if total > 0 else 0
    avg_time = sum(r["time_total"] for r in results if r["time_total"]) / total

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
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("RISULTATI")
    print("=" * 70)
    print(f"Config:    {full_config_name}")
    print(f"Accuracy:  {accuracy:.1f}% ({success}/{total})")
    print(f"Avg Time:  {avg_time:.2f}s")
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
