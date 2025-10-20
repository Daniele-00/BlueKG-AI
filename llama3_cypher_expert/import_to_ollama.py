"""
Script per convertire e importare il modello Llama3 fine-tuned in Ollama
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from google.cloud import storage
import os

# Configurazione
GCS_BUCKET = "llama3-tuning"
CHECKPOINT_PATH = "postprocess/node-0/checkpoints/final/checkpoint-final"
LOCAL_ADAPTER_PATH = "./llama3_adapter"
LOCAL_MERGED_PATH = "./llama3_merged"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def download_from_gcs(bucket_name, source_folder, destination_folder):
    """Scarica i file dal bucket GCS"""
    print(f"📥 Scaricando adapter da gs://{bucket_name}/{source_folder}...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)
    
    os.makedirs(destination_folder, exist_ok=True)
    
    for blob in blobs:
        if not blob.name.endswith('/'):
            file_path = os.path.join(destination_folder, os.path.basename(blob.name))
            blob.download_to_filename(file_path)
            print(f"  ✓ {os.path.basename(blob.name)}")
    
    print("✅ Download completato!\n")

def merge_and_save():
    """Merge del modello base con LoRA e salvataggio"""
    print("🔄 Caricamento modello base...")
    
    # Carica modello base
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",  # Usa CPU per il merge
        low_cpu_mem_usage=True
    )
    
    print("🔧 Applicazione adapter LoRA...")
    # Applica LoRA
    model = PeftModel.from_pretrained(model, LOCAL_ADAPTER_PATH)
    
    print("🔀 Merge dei pesi...")
    # Merge dei pesi
    model = model.merge_and_unload()
    
    print("💾 Salvataggio modello unificato...")
    os.makedirs(LOCAL_MERGED_PATH, exist_ok=True)
    model.save_pretrained(LOCAL_MERGED_PATH)
    tokenizer.save_pretrained(LOCAL_MERGED_PATH)
    
    print(f"✅ Modello salvato in: {LOCAL_MERGED_PATH}\n")
    return LOCAL_MERGED_PATH

def create_modelfile(model_path):
    """Crea il Modelfile per Ollama"""
    modelfile_content = f"""FROM {model_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM \"\"\"Sei un assistente esperto in Cypher e Neo4j.\"\"\"
"""
    
    modelfile_path = "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile creato: {modelfile_path}\n")
    return modelfile_path

def import_to_ollama():
    """Importa il modello in Ollama"""
    print("📦 Importazione in Ollama...")
    print("Esegui questi comandi:\n")
    print("cd " + os.getcwd())
    print("ollama create llama3-cypher -f Modelfile")
    print("\nDopo l'importazione, testa con:")
    print("ollama run llama3-cypher \"Come creo un nodo in Neo4j?\"\n")

def main():
    print("🦙 Llama3 Cypher Expert - Importazione in Ollama\n")
    print("=" * 60 + "\n")
    
    # Step 1: Scarica adapter (se non già presente)
    if not os.path.exists(LOCAL_ADAPTER_PATH):
        download_from_gcs(GCS_BUCKET, CHECKPOINT_PATH, LOCAL_ADAPTER_PATH)
    else:
        print("ℹ️  Adapter già scaricato.\n")
    
    # Step 2: Merge e salva modello unificato
    if not os.path.exists(LOCAL_MERGED_PATH):
        model_path = merge_and_save()
    else:
        model_path = LOCAL_MERGED_PATH
        print(f"ℹ️  Modello merged già presente in {model_path}\n")
    
    # Step 3: Crea Modelfile
    create_modelfile(model_path)
    
    # Step 4: Istruzioni per import in Ollama
    import_to_ollama()
    
    print("=" * 60)
    print("✅ Setup completato! Segui le istruzioni sopra per importare in Ollama.")

if __name__ == "__main__":
    main()
