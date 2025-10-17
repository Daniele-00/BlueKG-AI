import json
import requests
from neo4j import GraphDatabase
import csv
import sys

# =======================================================================
# CONFIGURAZIONE
# =======================================================================
API_URL = "http://127.0.0.1:8000"  # URL base della tua API chatbot

# --- INSERISCI QUI I TUOI DATI DI CONNESSIONE A NEO4J ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Blue2018"  # Aggiorna con la tua password

# =======================================================================


def compare_results(result1, result2) -> bool:
    """
    Confronta due risultati di query in modo robusto, ignorando i nomi degli alias
    e l'ordine delle righe.
    """
    if len(result1) != len(result2):
        return False

    if not result1:  # Se entrambe le liste sono vuote, sono uguali
        return True

    # --- NUOVA LOGICA: IGNORA GLI ALIAS ---
    try:
        # 1. Controlla che il numero di "colonne" sia lo stesso
        if len(result1[0]) != len(result2[0]):
            return False

        # 2. Estrai solo i valori da ogni riga (dizionario)
        # Esempio: {'totaleFatturato': 530} diventa (530,)
        values1 = {tuple(d.values()) for d in result1}
        values2 = {tuple(d.values()) for d in result2}

        # 3. Confronta i due set di tuple di valori
        return values1 == values2

    except Exception as e:
        print(f"  - ⚠️  Warning: fallback al confronto standard a causa di: {e}")
        # Fallback al metodo precedente in caso di dati non standard
        set1 = {tuple(sorted(d.items())) for d in result1}
        set2 = {tuple(sorted(d.items())) for d in result2}
        return set1 == set2


def run_query(tx, query, params=None):
    """Funzione helper per eseguire una query e raccogliere i risultati."""
    result = tx.run(query, params)
    return [r.data() for r in result]


def run_validation(model_name: str):
    """Esegue il set di validazione confrontando i risultati delle query."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Connessione a Neo4j stabilita.")
    except Exception as e:
        print(f"❌ Errore di connessione a Neo4j: {e}")
        return

    with open("validation_set.json", "r") as f:
        validation_data = json.load(f)

    success_count = 0
    failed_tests = []
    total_tests = len(validation_data)

    print(f"🚀 Avvio della validazione (basata sui risultati) su {total_tests} test...")

    comprehensive_results = []
    for i, test in enumerate(validation_data):
        question = test["question"]
        expected_cypher = test["expected_cypher"]
        user_id = f"validation_{test['id']}"
        generated_cypher = "N/A"  # Inizializziamo per sicurezza

        print(f"\n[{i+1}/{total_tests}] Test ID: {test['id']}...")
        print(f"  - Domanda: '{question}'")

        try:
            # Step 1: Resetta la memoria
            # requests.delete(f"{API_URL}/conversation/{user_id}")

            # Step 2: Ottieni la query generata
            response = requests.post(
                f"{API_URL}/ask", json={"question": question, "user_id": user_id}
            )
            response_data = response.json()
            generated_cypher = response_data.get("query_generata")
            timing_details = response_data.get("timing_details", {})
            if not generated_cypher:
                raise ValueError("La risposta dell'API non contiene 'query_generata'")

            # Step 3: Esegui e confronta i risultati
            with driver.session() as session:
                try:
                    generated_result = session.execute_read(run_query, generated_cypher)
                except Exception as e:
                    raise ValueError(f"La query GENERATA ha un errore di sintassi: {e}")

                expected_result = session.execute_read(run_query, expected_cypher)

                if compare_results(generated_result, expected_result):
                    success_count += 1
                    print("  - ✅ SUCCESSO (I risultati corrispondono)")
                else:
                    raise ValueError(
                        f"I risultati NON corrispondono.\n    - ATTESO: {expected_result}\n    - GENERATO: {generated_result}"
                    )
                comprehensive_results.append(
                    {
                        "model": model_name,
                        "test_id": test["id"],
                        "success": True,
                        "tempo_generazione_query": timing_details.get(
                            "generazione_query"
                        ),
                        "tempo_esecuzione_db": timing_details.get("esecuzione_db"),
                        "tempo_sintesi_risposta": timing_details.get(
                            "sintesi_risposta"
                        ),
                        "tempo_totale": timing_details.get("totale"),
                        "error": "",
                    }
                )

        except Exception as e:
            # --- MODIFICA 1: REGISTRAZIONE DELL'ERRORE PIÙ COMPLETA ---
            # Ora, indipendentemente dal motivo del fallimento, creiamo sempre
            # un dizionario con la stessa struttura.
            print(f"  - ❌ FALLITO: {e}")
            comprehensive_results.append(
                {
                    "model": model_name,
                    "test_id": test["id"],
                    "success": False,
                    "tempo_generazione_query": timing_details.get("generazione_query"),
                    "tempo_esecuzione_db": timing_details.get("esecuzione_db"),
                    "tempo_sintesi_risposta": timing_details.get("sintesi_risposta"),
                    "tempo_totale": timing_details.get("totale"),
                    "error": str(e),
                }
            )
    driver.close()

    # --- NUOVO: Salva tutti i risultati in un unico file CSV ---
    csv_file = "validation_comprehensive_results.csv"
    csv_columns = [
        "model",
        "test_id",
        "success",
        "tempo_generazione_query",
        "tempo_esecuzione_db",
        "tempo_sintesi_risposta",
        "tempo_totale",
        "error",
    ]
    try:
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            # Scrivi l'header solo se il file è nuovo/vuoto
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerows(comprehensive_results)
        print(f"\n✅ Risultati dettagliati aggiunti a {csv_file}")
    except IOError:
        print(f"❌ Errore durante il salvataggio del file CSV.")
    # Stampa il report finale
    accuracy = (success_count / total_tests) * 100 if total_tests > 0 else 0
    print("\n" + "=" * 50)
    print("📊 REPORT DI VALIDAZIONE FINALE 📊")
    print("=" * 50)
    print(
        f"Accuratezza Totale: {accuracy:.2f}% ({success_count}/{total_tests} superati)"
    )

    if failed_tests:
        print("\n🔬 Dettaglio Test Falliti:")
        # --- MODIFICA 2: STAMPA DEL REPORT PIÙ ROBUSTA ---
        # Il report finale è stato modificato per stampare il messaggio di errore specifico,
        # che è molto più utile per il debugging.
        for failure in failed_tests:
            print("\n" + "-" * 20)
            print(f"ID Test: {failure['id']}")
            print(f"Domanda: {failure['question']}")
            print(f"Query ATTESA   : {failure['expected_cypher']}")
            print(f"Query GENERATA : {failure['generated_cypher']}")
            print(f"MOTIVO FALLIMENTO: {failure['error_message']}")
            print("-" * 20)

    # Salva i risultati in un file JSON
    results_summary = {
        "total_tests": total_tests,
        "success": success_count,
        "failures": len(failed_tests),
    }

    with open("validation_results.json", "w") as f:
        json.dump(results_summary, f, indent=4)

    print("\n✅ Risultati di validazione salvati in validation_results.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Errore: Specifica il nome del modello come argomento.")
        print("Esempio: python3 run_validation_by_results.py gpt-4o-mini")
        sys.exit(1)

    model_under_test = sys.argv[1]
    run_validation(model_under_test)
