# FILE: openclip_worker.py (Đã sửa đổi)
import os, torch, open_clip
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# --- OpenCLIP Setup ---
DEVICE = os.getenv("OPENCLIP_DEVICE", "cuda:1") # Sử dụng biến môi trường cho device
MODEL_ARCH = "ViT-H-14"
PRETRAINED_DATASET = "laion2b_s32b_b79k" # Tải model chuẩn từ hub

app = FastAPI()
model_data = {}

@app.on_event("startup")
def load_model():
    print(f"--- OpenCLIP Worker: Đang tải model '{MODEL_ARCH}' với weights '{PRETRAINED_DATASET}' lên {DEVICE}... ---")
    
    # Tải model và các thành phần cần thiết từ open_clip
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=MODEL_ARCH, 
        pretrained=PRETRAINED_DATASET
    )
    
    model_data['model'] = model.to(DEVICE).eval()
    model_data['transform'] = transform
    model_data['tokenizer'] = open_clip.get_tokenizer(MODEL_ARCH)
    print("--- OpenCLIP Worker: Model đã được tải và sẵn sàng. ---")

@app.post("/embed")
async def get_embedding(text_query: str = Form(None), image_file: UploadFile = File(None)):
    if not text_query and not image_file:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp text_query hoặc image_file")
    
    vec = None
    with torch.no_grad():
        if image_file:
            image_bytes = await image_file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            tensor = model_data['transform'](image).unsqueeze(0).to(DEVICE)
            vec_tensor = model_data['model'].encode_image(tensor)
            vec_tensor /= vec_tensor.norm(p=2, dim=-1, keepdim=True)
            vec = vec_tensor.cpu().numpy().tolist()
        elif text_query:
            tokens = model_data['tokenizer']([text_query]).to(DEVICE)
            vec_tensor = model_data['model'].encode_text(tokens)
            vec_tensor /= vec_tensor.norm(p=2, dim=-1, keepdim=True)
            vec = vec_tensor.cpu().numpy().tolist()
            
    return {"embedding": vec}