import json
import yaml
from pathlib import Path
from typing import Dict, List
import logging
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Usa i tuoi modelli esistenti
from chatbot_specialist import (
    MODELLI_STRATIFICATI,
    run_specialist_router_agent,
    run_translator_agent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: Classifica domande per specialist
# ============================================================================


def classify_questions(input_file: str) -> Dict[str, List[Dict]]:
    """Classifica ogni domanda usando il Router"""

    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    classified = {
        "SALES_CYCLE": [],
        "PURCHASE_CYCLE": [],
        "GENERAL_QUERY": [],
        "DITTE_QUERY": [],
        "CROSS_DOMAIN": [],
    }

    logger.info(f"📊 Classificazione di {len(questions)} domande...")

    for i, q in enumerate(questions, 1):
        # Traduci (il router vuole inglese)
        english_task = run_translator_agent(q["input"])

        # Classifica
        route = run_specialist_router_agent(english_task)

        # Aggiungi alla categoria giusta
        example = {
            "id": f"{route.lower()}_{i:03d}",
            "question": q["input"],
            "cypher": q["output"],
        }

        if route in classified:
            classified[route].append(example)
            logger.info(f"  [{i}/{len(questions)}] {route}: {q['input'][:60]}...")
        else:
            logger.warning(f"  ⚠️ Route sconosciuto: {route}")
            classified["GENERAL_QUERY"].append(example)  # Fallback

    # Stats
    logger.info("\n📊 Risultati classificazione:")
    for specialist, examples in classified.items():
        logger.info(f"  {specialist}: {len(examples)} esempi")

    return classified


# ============================================================================
# STEP 2: Genera varianti (opzionale ma utile)
# ============================================================================

VARIANT_PROMPT = """
Generate 2 paraphrased variants of this Italian question. Keep the same meaning but change wording.

Original: {question}

Output format (one per line, no numbers):
Variant 1
Variant 2

Only output the 2 variants, nothing else.
"""


def generate_variants(question: str, n: int = 2) -> List[str]:
    """Genera n varianti della domanda usando LLM"""
    try:
        prompt = PromptTemplate.from_template(VARIANT_PROMPT)
        chain = prompt | MODELLI_STRATIFICATI["translator"] | StrOutputParser()

        result = chain.invoke({"question": question})

        # Parse variants
        variants = [line.strip() for line in result.split("\n") if line.strip()]
        return variants[:n]

    except Exception as e:
        logger.warning(f"⚠️ Errore generazione varianti: {e}")
        return []


def expand_examples_with_variants(
    classified: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """Espande ogni esempio con 2 varianti"""

    expanded = {k: [] for k in classified.keys()}

    logger.info("\n🔄 Generazione varianti...")

    for specialist, examples in classified.items():
        logger.info(f"\n  {specialist}:")

        for ex in examples:
            # Esempio originale
            expanded[specialist].append(ex)

            # Genera varianti
            variants = generate_variants(ex["question"], n=2)

            for i, variant in enumerate(variants, 1):
                variant_ex = {
                    "id": f"{ex['id']}_var{i}",
                    "question": variant,
                    "cypher": ex["cypher"],  # Stessa query
                }
                expanded[specialist].append(variant_ex)

            logger.info(f"    {ex['id']}: +{len(variants)} varianti")

    # Stats
    logger.info("\n📊 Esempi totali (con varianti):")
    for specialist, examples in expanded.items():
        logger.info(f"  {specialist}: {len(examples)} esempi")

    return expanded


# ============================================================================
# STEP 3: Esporta in YAML
# ============================================================================


def export_to_yaml(classified: Dict[str, List[Dict]], output_dir: str = "examples"):
    """Esporta esempi in file YAML separati"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    logger.info(f"\n💾 Esportazione in {output_dir}/...")

    for specialist, examples in classified.items():
        if not examples:
            logger.warning(f"  ⚠️ {specialist}: nessun esempio, skip")
            continue

        file_path = output_path / f"{specialist}.yaml"

        data = {"examples": examples}

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

        logger.info(f"  ✅ {file_path}: {len(examples)} esempi")

    logger.info("\n✅ Esportazione completata!")


# ============================================================================
# MAIN
# ============================================================================


def main():
    INPUT_FILE = "validation_set_2.json"
    OUTPUT_DIR = "examples"
    GENERATE_VARIANTS = True  # Cambia in False se vuoi solo classificare

    logger.info("🚀 Inizio organizzazione esempi\n")

    # STEP 1: Classifica
    classified = classify_questions(INPUT_FILE)

    # STEP 2: Genera varianti (opzionale)
    if GENERATE_VARIANTS:
        classified = expand_examples_with_variants(classified)

    # STEP 3: Esporta
    export_to_yaml(classified, OUTPUT_DIR)

    logger.info("\n🎉 Done! Controlla la cartella examples/")


if __name__ == "__main__":
    main()
