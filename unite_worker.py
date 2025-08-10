# FILE: unite_worker.py (Sửa lỗi Import cho Uvicorn)

import os
import sys
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from io import BytesIO
import time

# --- [SỬA LỖI CHO UVICORN] Thêm đường dẫn của file này vào sys.path ---
# Điều này đảm bảo rằng các module trong cùng thư mục sẽ được tìm thấy
# bất kể Uvicorn được chạy từ đâu.
# __file__ là một biến đặc biệt chứa đường dẫn đến file script hiện tại.
WORKER_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKER_FILE_DIR not in sys.path:
    sys.path.insert(0, WORKER_FILE_DIR)

try:
    # Bây giờ, việc import sẽ thành công vì thư mục chứa các file này
    # đã được thêm vào sys.path ở trên.
    from qwen_vl_utils import process_vision_info
    from modeling_unite import UniteQwen2VL
    from transformers import AutoTokenizer, AutoProcessor
    print(f"--- Unite Worker: Đã import thành công các module helper từ '{WORKER_FILE_DIR}'. ---")
except ImportError as e:
    print(f"!!! LỖI IMPORT NGHIÊM TRỌNG !!!")
    print(f"Không thể import các file helper của Unite. Lỗi: {e}")
    print(f"Hãy đảm bảo 2 file 'qwen_vl_utils.py' và 'modeling_unite.py' nằm trong cùng thư mục với file worker này: '{WORKER_FILE_DIR}'.")
    sys.exit(1)
# -----------------------------------------------------------------

Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
MODEL_NAME = "friedrichor/Unite-Base-Qwen2-VL-2B"
DEVICE = os.getenv("UNITE_DEVICE", "cuda:0") 
DTYPE = torch.float16

# --- Lifespan Event Handler (Thay cho on_event deprecated) ---
from contextlib import asynccontextmanager

model_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code chạy khi khởi động (startup)
    print(f"--- Unite Worker: Loading model '{MODEL_NAME}' onto {DEVICE}... ---")
    st_load = time.time()
    
    device = torch.device(DEVICE)
    model = UniteQwen2VL.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=DTYPE,
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, tokenizer=tokenizer)

    model_data['model'] = model
    model_data['processor'] = processor
    model_data['device'] = device
    
    print(f"--- Unite Worker: Model loaded in {time.time() - st_load:.2f}s. Ready. ---")
    
    yield
    
    # Code chạy khi tắt (shutdown)
    print("--- Unite Worker: Shutting down. Clearing model data. ---")
    model_data.clear()

app = FastAPI(lifespan=lifespan)

# --- Helper Functions (Không đổi) ---
def process_messages(msg, processor, device):
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(msg)
    inputs = processor(
        text=[text], 
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)

def embed_image_func(image, model, processor, device):
    messages_img = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "\nSummary above image in one word:"},
        ]}
    ]
    inputs_img = process_messages(messages_img, processor, device)
    return model(**inputs_img)

def embed_text_func(text, model, processor, device):
    messages_txt = [
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "text", "text": "\nSummary above sentence in one word:"},
        ]}
    ]
    inputs_text = process_messages(messages_txt, processor, device)
    return model(**inputs_text)


@app.post("/embed")
async def get_embedding(text_query: str = Form(None), image_file: UploadFile = File(None)):
    """
    Generate an embedding for either a text query or an uploaded image.
    """
    if not text_query and not image_file:
        raise HTTPException(status_code=400, detail="Please provide 'text_query' or 'image_file'")

    model = model_data.get('model')
    processor = model_data.get('processor')
    device = model_data.get('device')

    if not all([model, processor, device]):
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    vec = None
    with torch.no_grad():
        if image_file:
            image_bytes = await image_file.read()
            image = Image.open(BytesIO(image_bytes))
            vec_tensor = embed_image_func(image, model, processor, device)
            vec = vec_tensor.cpu().float().numpy().tolist()
            
        elif text_query:
            vec_tensor = embed_text_func(text_query, model, processor, device)
            vec = vec_tensor.cpu().float().numpy().tolist()

    if vec is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

    return {"embedding": vec}

# Dành cho việc chạy test trực tiếp
if __name__ == "__main__":
    import uvicorn
    # Chạy server trên port 8003, cho phép truy cập từ mọi địa chỉ
    uvicorn.run(app, host="0.0.0.0", port=8003)