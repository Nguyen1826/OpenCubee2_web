# FILE: openclip_worker.py
import os, torch, open_clip
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# --- OpenCLIP Setup ---
DEVICE = os.getenv("OPENCLIP_DEVICE", "cuda:1") # Sử dụng biến môi trường cho device
MODEL_ARCH = "ViT-H-14"

# --- THAY ĐỔI Ở ĐÂY: Trỏ đến file model cục bộ ---
# Thay vì tải từ internet, chúng ta sẽ chỉ định đường dẫn tới file weights
# Hãy chắc chắn rằng file model của bạn (.bin hoặc .safetensors) nằm trong thư mục này
MODEL_WEIGHTS_PATH = "weights/cliph14/open_clip_pytorch_model.bin" 
# Nếu bạn dùng file .safetensors, hãy đổi tên file ở trên thành "open_clip_model.safetensors"

app = FastAPI()
model_data = {}

@app.on_event("startup")
def load_model():
    # Thêm một bước kiểm tra để đảm bảo file tồn tại trước khi load
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        error_msg = f"LỖI NGHIÊM TRỌNG: Không tìm thấy file model tại '{MODEL_WEIGHTS_PATH}'"
        print(error_msg)
        # Gây ra lỗi để ứng dụng FastAPI không khởi động được nếu không có model
        raise FileNotFoundError(error_msg)

    print(f"--- OpenCLIP Worker: Đang tải model '{MODEL_ARCH}' từ file cục bộ '{MODEL_WEIGHTS_PATH}' lên {DEVICE}... ---")
    
    # --- THAY ĐỔI Ở ĐÂY: Sử dụng đường dẫn cục bộ trong tham số 'pretrained' ---
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=MODEL_ARCH, 
        pretrained=MODEL_WEIGHTS_PATH  # Sử dụng đường dẫn file thay vì tên model trên hub
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
        elif text_query: # Sửa lại logic để không bị lỗi nếu cả hai đều None
            tokens = model_data['tokenizer']([text_query]).to(DEVICE)
            vec_tensor = model_data['model'].encode_text(tokens)
            vec_tensor /= vec_tensor.norm(p=2, dim=-1, keepdim=True)
            vec = vec_tensor.cpu().numpy().tolist()
            
    return {"embedding": vec}