import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from google import genai
from google.genai import types
from io import BytesIO
import uvicorn

# --- Cấu hình ---
# Lấy API key từ biến môi trường hoặc hardcode (không khuyến khích)
# !!! THAY API KEY CỦA BẠN VÀO ĐÂY NẾU KHÔNG DÙNG BIẾN MÔI TRƯỜNG !!!
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD--J-Roa8srsd2QBey7PVfbyxhbGmCrLM") 
if not API_KEY:
    raise ValueError("Vui lòng cung cấp Gemini API Key.")

# Khởi tạo client giống hệt như code của bạn
client = genai.Client(api_key=API_KEY)
app = FastAPI()

class ImageGenRequest(BaseModel):
    query: str


# --- Logic sinh ảnh (Được điều chỉnh để khớp 100% với code của bạn) ---
def generate_image_data(query: str) -> bytes:
    """
    Sinh ảnh từ query và trả về dưới dạng bytes, sử dụng cú pháp bạn đã cung cấp.
    """
    try:
        # 1. Tạo prompt
        contents = (f'Hi, can you create a picture having this content for image to image retrieval: {query}')
        
        # 2. Tạo config với response_modalities - đây là phần quan trọng
        config = types.GenerateContentConfig(
			response_modalities=['TEXT', 'IMAGE'] 
		)

        # 3. Gọi API bằng client.models.generate_content
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation", # Sử dụng đúng tên model
            contents=contents,
            config=config
        )
    
        # 4. Trích xuất dữ liệu ảnh từ response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Tìm phần chứa dữ liệu ảnh
                if part.inline_data is not None:
                    # Trả về dữ liệu bytes của ảnh
                    return part.inline_data.data
        
        # Nếu vòng lặp kết thúc mà không tìm thấy ảnh, ném lỗi
        raise ValueError("Không tìm thấy dữ liệu ảnh trong phản hồi từ API. Phản hồi có thể chỉ chứa text.")

    except Exception as e:
        print(f"Lỗi khi gọi API của Gemini: {e}")
        # Ném lại lỗi để endpoint có thể bắt và trả về lỗi 500
        raise

# --- Endpoint ---
@app.post("/generate")
async def handle_image_generation(request: ImageGenRequest):
    """
    Endpoint nhận query text và trả về ảnh.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query không được để trống.")
    
    try:
        # Gọi hàm logic đã được sửa đổi
        image_bytes = generate_image_data(request.query)
        # Trả về dữ liệu ảnh thô với content type là image/png
        return Response(content=image_bytes, media_type="image/png")
    except ValueError as e: # Bắt lỗi cụ thể khi không tìm thấy ảnh
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e: # Bắt các lỗi chung khác
        raise HTTPException(status_code=500, detail=f"Lỗi phía server khi sinh ảnh: {str(e)}")


if __name__ == "__main__":
    # Chạy service này trên một port khác, ví dụ 8004
    uvicorn.run(app, host="0.0.0.0", port=8004)