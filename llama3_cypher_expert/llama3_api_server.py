"""
Server FastAPI per servire il modello Llama3 fine-tuned con API compatibile OpenAI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from google.cloud import storage
import os
import uvicorn

app = FastAPI(title="Llama3 Cypher Expert API")

# Configurazione
GCS_BUCKET = "llama3-tuning"
CHECKPOINT_PATH = "postprocess/node-0/checkpoints/final/checkpoint-final"
LOCAL_MODEL_PATH = "./llama3_finetuned"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Variabili globali per il modello
model = None
tokenizer = None


# Modelli di richiesta compatibili con OpenAI
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama3-cypher"
    messages: List[Message]
    temperature: float = 0.3
    max_tokens: int = 512
    top_p: float = 0.9
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int = 0
    model: str = "llama3-cypher"
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def download_from_gcs(bucket_name: str, source_folder: str, destination_folder: str):
    """Scarica i file dal bucket GCS"""
    print(f"📥 Scaricando modello da gs://{bucket_name}/{source_folder}...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)

    os.makedirs(destination_folder, exist_ok=True)

    for blob in blobs:
        if not blob.name.endswith("/"):
            file_path = os.path.join(destination_folder, os.path.basename(blob.name))
            blob.download_to_filename(file_path)
            print(f"  ✓ Scaricato: {os.path.basename(blob.name)}")

    print("✅ Download completato!\n")


def load_model():
    """Carica il modello all'avvio del server"""
    global model, tokenizer

    print("🚀 Avvio caricamento modello...")

    # Scarica adapter se necessario (solo 168 MB!)
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_from_gcs(GCS_BUCKET, CHECKPOINT_PATH, LOCAL_MODEL_PATH)

    print("🔄 Caricamento modello base (quantizzato 4-bit per risparmiare spazio)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Carica con quantizzazione 4-bit - risparmia ~70% spazio!
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("🔧 Applicazione adapter LoRA...")
    model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
    model.eval()

    print("✅ Modello caricato e pronto!")
    print("💾 Spazio totale usato: ~5 GB (modello 4-bit + adapter)\n")


def format_messages(messages: List[Message]) -> str:
    """Formatta i messaggi nel formato Llama 3.1 Instruct"""
    formatted = "<|begin_of_text|>"

    for msg in messages:
        if msg.role == "system":
            formatted += (
                f"<|start_header_id|>system<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            )
        elif msg.role == "user":
            formatted += (
                f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            )
        elif msg.role == "assistant":
            formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"

    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted


@app.on_event("startup")
async def startup_event():
    """Carica il modello all'avvio del server"""
    load_model()


@app.get("/")
async def root():
    return {"message": "Llama3 Cypher Expert API - Online", "status": "ready"}


@app.get("/v1/models")
async def list_models():
    """Endpoint compatibile OpenAI per listare i modelli"""
    return {
        "object": "list",
        "data": [
            {
                "id": "llama3-cypher",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint compatibile OpenAI per chat completions"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Formatta i messaggi
        prompt = format_messages(request.messages)

        # Tokenizza
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Genera
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decodifica
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Estrae solo la risposta dell'assistant
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response_text = full_response.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1]
            response_text = response_text.split("<|eot_id|>")[0].strip()
        else:
            response_text = full_response

        # Calcola token usage
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(outputs[0]) - input_tokens

        return ChatCompletionResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🦙 Llama3 Cypher Expert API Server")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
