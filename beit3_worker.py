# FILE: beit3_worker.py
import os, sys, torch
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from torchvision import transforms

# --- BEiT-3 Setup ---
# Assumes this script is run from the project root
script_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(script_dir, 'scripts'))
try:
    from modeling_finetune import BEiT3ForRetrieval, _get_large_config
    from datasets import get_sentencepiece_model_for_beit3
except ImportError as e:
    sys.exit(f"FATAL: BEiT-3 source files not found. Make sure they are in the 'scripts' directory. Error: {e}")

DEVICE = os.getenv("BEIT3_DEVICE", "cuda:0") # Use environment variable for device
MODEL_PATH = "./weights/beit3/beit3_large_patch16_384_coco_retrieval.pth"
SPM_PATH = "./weights/beit3/beit3.spm"

app = FastAPI()
model_data = {}

@app.on_event("startup")
def load_model():
    print(f"--- BEiT-3 Worker: Loading model onto {DEVICE}... ---")
    model_config = _get_large_config(img_size=384)
    model = BEiT3ForRetrieval(model_config)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt.get('module')), strict=False)
    model_data['model'] = model.to(DEVICE).eval()
    model_data['tokenizer'] = get_sentencepiece_model_for_beit3(SimpleNamespace(sentencepiece_model=SPM_PATH))
    model_data['transform'] = transforms.Compose([
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    print("--- BEiT-3 Worker: Model loaded and ready. ---")

@app.post("/embed")
async def get_embedding(text_query: str = Form(None), image_file: UploadFile = File(None)):
    if not text_query and not image_file:
        raise HTTPException(status_code=400, detail="Provide text_query or image_file")
    vec = None
    with torch.no_grad():
        if image_file:
            image = Image.open(BytesIO(await image_file.read())).convert("RGB")
            tensor = model_data['transform'](image).unsqueeze(0).to(DEVICE)
            vec_tensor, _ = model_data['model'](image=tensor, only_infer=True)
            vec = vec_tensor.cpu().numpy().tolist()
        else: # text_query
            tokenizer = model_data['tokenizer']
            tokens = tokenizer.encode(text_query, out_type=int)[:64]
            padded = tokens + [tokenizer.pad_token_id] * (64 - len(tokens))
            text_input = torch.tensor([padded], device=DEVICE)
            padding_mask = (text_input == tokenizer.pad_token_id)
            dummy_image = torch.zeros(1, 3, 384, 384, device=DEVICE)
            _, vec_tensor = model_data['model'](image=dummy_image, text_description=text_input, padding_mask=padding_mask, only_infer=True)
            vec = vec_tensor.cpu().numpy().tolist()
    return {"embedding": vec}
