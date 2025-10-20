import json
import random
import math  # Importa il modulo math per usare ceil

# --- Configurazione ---
INPUT_DATASET_FILE = "training_data_vertex_chat.jsonl"  # Il file .jsonl con i messaggi (system/user/assistant)
TRAIN_OUTPUT_FILE = "train_set.jsonl"
VALIDATION_OUTPUT_FILE = "validation_set.jsonl"
VALIDATION_SPLIT_RATIO = 0.2  # 20% per la validazione, 80% per il training

# --- 1. Carica tutte le righe dal dataset ---
print(f"Caricamento del dataset da '{INPUT_DATASET_FILE}'...")
lines = []
try:
    with open(INPUT_DATASET_FILE, "r", encoding="utf-8") as f_in:
        for line in f_in:
            try:
                # Tentativo di caricare ogni riga come JSON per assicurarsi che sia valida
                json.loads(line)
                lines.append(line.strip())  # Aggiungi la riga (come stringa) alla lista
            except json.JSONDecodeError:
                print(f"ATTENZIONE: Riga non JSON valida saltata: {line.strip()}")
except FileNotFoundError:
    print(f"ERRORE: File di input '{INPUT_DATASET_FILE}' non trovato!")
    exit()
except Exception as e:
    print(f"ERRORE durante la lettura del file: {e}")
    exit()

if not lines:
    print(
        "ERRORE: Nessuna riga valida trovata nel file di input. Impossibile continuare."
    )
    exit()

print(f"Caricate {len(lines)} righe valide.")

# --- 2. Mischia le righe ---
print("Mescolamento del dataset...")
random.shuffle(lines)
print("Dataset mescolato.")

# --- 3. Calcola il punto di divisione ---
total_lines = len(lines)
# math.ceil arrotonda all'intero superiore per assicurarsi che il validation set abbia almeno la % richiesta
validation_size = math.ceil(total_lines * VALIDATION_SPLIT_RATIO)
train_size = total_lines - validation_size

print(f"Dimensione totale: {total_lines}")
print(f"Dimensione Training Set ({100*(1-VALIDATION_SPLIT_RATIO):.0f}%): {train_size}")
print(
    f"Dimensione Validation Set ({100*VALIDATION_SPLIT_RATIO:.0f}%): {validation_size}"
)

# --- 4. Scrivi i file di output ---
print(f"Scrittura del Training Set in '{TRAIN_OUTPUT_FILE}'...")
try:
    with open(TRAIN_OUTPUT_FILE, "w", encoding="utf-8") as f_train:
        for i in range(train_size):
            f_train.write(lines[i] + "\n")  # Scrivi le prime N righe mischiate
except Exception as e:
    print(f"ERRORE durante la scrittura del file di training: {e}")
    exit()

print(f"Scrittura del Validation Set in '{VALIDATION_OUTPUT_FILE}'...")
try:
    with open(VALIDATION_OUTPUT_FILE, "w", encoding="utf-8") as f_val:
        # Scrivi le righe rimanenti (dalla posizione train_size fino alla fine)
        for i in range(train_size, total_lines):
            f_val.write(lines[i] + "\n")
except Exception as e:
    print(f"ERRORE durante la scrittura del file di validazione: {e}")
    exit()

print("\n✅ Operazione completata!")
print(
    f"File creati: '{TRAIN_OUTPUT_FILE}' ({train_size} righe) e '{VALIDATION_OUTPUT_FILE}' ({validation_size} righe)."
)
