# FILE: bge_worker.py
import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import AutoModel
from PIL import Image
from io import BytesIO
import time

# --- Configuration ---
MODEL_NAME = "BAAI/BGE-VL-large"
# Use environment variable for device, fallback to cuda:0
DEVICE = os.getenv("BGE_DEVICE", "cuda:0") 
DTYPE = torch.float16

app = FastAPI()
model_data = {}

@app.on_event("startup")
def load_model():
    """
    Load the BGE-VL-Large model on startup.
    """
    print(f"--- BGE Worker: Loading model '{MODEL_NAME}' onto {DEVICE}... ---")
    st_load = time.time()
    
    device = torch.device(DEVICE)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).to(device, dtype=DTYPE).eval()
    
    # The set_processor call is crucial for this model
    model.set_processor(MODEL_NAME)

    if hasattr(torch, 'compile'):
        print("--- BGE Worker: Compiling model with torch.compile()... ---")
        model = torch.compile(model)

    model_data['model'] = model
    model_data['device'] = device
    
    print(f"--- BGE Worker: Model loaded and compiled in {time.time() - st_load:.2f}s. Ready. ---")

@app.post("/embed")
async def get_embedding(text_query: str = Form(None), image_file: UploadFile = File(None)):
    """
    Generate an embedding for either a text query or an uploaded image.
    """
    if not text_query and not image_file:
        raise HTTPException(status_code=400, detail="Please provide 'text_query' or 'image_file'")

    model = model_data.get('model')
    if not model:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    vec = None
    with torch.no_grad():
        if image_file and text_query:
            # SỬA DÒNG NÀY: Thêm dấu ngoặc () sau .read
            image_bytes = await image_file.read()  # <--- SỬA LỖI Ở ĐÂY
            # The BGE model expects a list of PIL Images or BytesIO objects
            vec_tensor = model.encode(images=[BytesIO(image_bytes)], text=[text_query])
            vec = vec_tensor.cpu().numpy().tolist()

        elif image_file:
            # SỬA DÒNG NÀY: Thêm dấu ngoặc () sau .read
            image_bytes = await image_file.read() # <--- SỬA LỖI Ở ĐÂY
            # BGE model's encode function can take a list of PIL images or BytesIO
            vec_tensor = model.encode(images=[BytesIO(image_bytes)])
            vec = vec_tensor.cpu().numpy().tolist()
            
        elif text_query:
            # BGE model's encode function can take text
            # Sửa đổi nhỏ để nhất quán: truyền vào dạng list
            vec_tensor = model.encode(text=[text_query]) # <--- Sửa nhỏ cho nhất quán
            vec = vec_tensor.cpu().numpy().tolist()

    if vec is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

    return {"embedding": vec}