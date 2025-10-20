import json
import yaml  # Potrebbe servire 'pip install PyYAML'

# --- Configurazione ---
BASE_RULES_FILE = "/opt/BlueKG/training_set/base_cypher_rules_combiante.yaml"
INPUT_DATASET_FILE = "/opt/BlueKG/training_set/training_dataset.jsonl"  # Il tuo file con instruction/input/output
OUTPUT_DATASET_FILE = "training_data_vertex_chat.jsonl"

# --- Carica le Base Rules UNA VOLTA SOLA ---
try:
    with open(BASE_RULES_FILE, "r", encoding="utf-8") as f:
        # Assumendo che il tuo YAML abbia la chiave 'content:'
        config = yaml.safe_load(f)
        BASE_RULES_CONTENT = config.get("content", "").strip()
        if not BASE_RULES_CONTENT:
            print(f"ERRORE: 'content' non trovato o vuoto in {BASE_RULES_FILE}")
            exit()
except FileNotFoundError:
    print(f"ERRORE: File base rules '{BASE_RULES_FILE}' non trovato!")
    exit()
except Exception as e:
    print(f"ERRORE durante la lettura del file YAML '{BASE_RULES_FILE}': {e}")
    exit()

# --- Definisci lo Schema Placeholder (come prima) ---
SCHEMA_PLACEHOLDER_FOR_TRAINING = """--- SCHEMA DEL GRAFO ---
(Schema dettagliato fornito a runtime. Nodi principali: Cliente, Articolo, Documento, Fornitore, Ditta, Famiglia, Luogo, GruppoFornitore, Sottofamiglia, RigaDocumento, DocType. Relazioni principali: RAGGRUPPATO_SOTTO, APPARTIENE_A, SI_TROVA_A, HA_RICEVUTO, CONTIENE_RIGA, RIGUARDA_ARTICOLO, HA_EMESSO, HAS_ADDRESS, IS_TYPE)
"""

# --- Template del System Prompt (Include Regole Complete + Placeholder Schema) ---
# Nota: Usiamo f-string formattate correttamente
SYSTEM_PROMPT_CONTENT = f"""Sei un esperto Cypher. Genera solo codice. Obbedisci rigorosamente alle REGOLE e usa solo lo SCHEMA fornito.

--- REGOLE ---
{BASE_RULES_CONTENT}

{SCHEMA_PLACEHOLDER_FOR_TRAINING}
""".strip()  # .strip() per rimuovere spazi extra all'inizio/fine

print("Regole caricate e System Prompt Template creato.")

# --- Processa il Dataset ---
count = 0
skipped_count = 0

try:
    with open(INPUT_DATASET_FILE, "r", encoding="utf-8") as f_in, open(
        OUTPUT_DATASET_FILE, "w", encoding="utf-8"
    ) as f_out:

        for line_num, line in enumerate(f_in, 1):
            try:
                # Carica ogni riga JSON
                record = json.loads(line)

                # Estrai input (domanda) e output (query)
                # Adattalo se le chiavi nel tuo file sono diverse
                user_content = record.get("input")
                assistant_content = record.get("output")
                instruction_info = record.get(
                    "instruction", f"Riga {line_num}"
                )  # Per logging

                # Controllo di base
                if not user_content or not assistant_content:
                    print(
                        f"ATTENZIONE: Record saltato (riga {line_num}) per 'input' o 'output' mancante: {instruction_info}"
                    )
                    skipped_count += 1
                    continue

                # Crea l'esempio in formato chat
                chat_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_CONTENT,
                        },  # SEMPRE LO STESSO
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }

                # Scrivi la riga nel nuovo file JSONL
                f_out.write(json.dumps(chat_example, ensure_ascii=False) + "\n")
                count += 1

            except json.JSONDecodeError:
                print(f"ATTENZIONE: Riga {line_num} non è JSON valido, saltata.")
                skipped_count += 1
            except Exception as e:
                print(f"ERRORE inaspettato alla riga {line_num}: {e}")
                skipped_count += 1

except FileNotFoundError:
    print(f"ERRORE: File di input '{INPUT_DATASET_FILE}' non trovato!")
    exit()
except Exception as e:
    print(f"ERRORE durante l'apertura/scrittura dei file: {e}")
    exit()


print(f"\n✅ Fatto! Creato '{OUTPUT_DATASET_FILE}' con {count} esempi.")
if skipped_count > 0:
    print(f"⚠️ Saltati {skipped_count} record per errori o dati mancanti.")
