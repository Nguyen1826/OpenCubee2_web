# FILE: bge_m3_worker.py

import os
import sys
from fastapi import FastAPI, Form, HTTPException
import ollama
import time

# --- Configuration ---
MODEL_NAME = "bge-m3:567m"

app = FastAPI()

@app.on_event("startup")
def check_model():
    """
    Check if the Ollama model is available on startup.
    """
    print(f"--- BGE-M3 Worker: Checking for Ollama model '{MODEL_NAME}'... ---")
    st_check = time.time()
    try:
        ollama.show(MODEL_NAME)
        print(f"--- BGE-M3 Worker: Found model '{MODEL_NAME}'. Ready. Check took {time.time() - st_check:.2f}s. ---")
    except Exception as e:
        sys.exit(f"FATAL: Ollama model '{MODEL_NAME}' not found. Please run 'ollama pull {MODEL_NAME}' first. Error: {e}")

@app.post("/embed")
async def get_embedding(text_query: str = Form(None)):
    """
    Generate an embedding for a text query using Ollama.
    This worker only supports text queries based on the provided embed/retrieve scripts.
    """
    if not text_query:
        raise HTTPException(status_code=400, detail="Please provide 'text_query'. Image input is not supported by this worker.")

    try:
        response = ollama.embed(model=MODEL_NAME, input=text_query)
        vec = [response['embedding']] # ollama.embed returns a single embedding, we wrap it in a list to match other workers
    except Exception as e:
        print(f"ERROR calling Ollama: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding with Ollama. Error: {e}")

    return {"embedding": vec}
