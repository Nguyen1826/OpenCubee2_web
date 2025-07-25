# FILE: gateway_server.py (Final Version - WITH ENHANCED DEBUGGING)

import os
import sys
import traceback  # <-- THÊM THƯ VIỆN NÀY ĐỂ IN LỖI ĐẦY ĐỦ
import base64
import asyncio
from collections import defaultdict
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pymilvus import Collection, connections, utility

# --- Kỹ thuật xử lý import mà không cần __init__.py ---

# 1. Tự động xác định đường dẫn đến thư mục .../Repo
COMMON_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Thêm thư mục cha chung này vào sys.path của Python.
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)

# --- THAY ĐỔI QUAN TRỌNG: NÂNG CẤP KHỐI DEBUG ---
# In ra sys.path để chắc chắn rằng chúng ta đã thêm đúng đường dẫn
print("--- DEBUG: Python Search Paths (sys.path) ---")
for path in sys.path:
    print(f"  - {path}")
print("--------------------------------------------")

try:
    from Cubi.translate_query import translate_text
    print("--- Gateway Server: Successfully imported 'translate_text' from Cubi subfolder. ---")
except ImportError as e:
    # Đây là phần debug được nâng cấp
    print("\n" + "="*80)
    print("!!! FATAL IMPORT ERROR: Could not import 'translate_text'. !!!")
    print(f"--- The original error message was: ---> {e} <---")
    print("\n--- This error likely originates from a missing library inside 'translate_query.py' or one of its imports (like 'Enn.llm.main').")
    print("--- Detailed Traceback (shows exactly which file and line failed): ---")
    traceback.print_exc() # <-- In ra toàn bộ dấu vết lỗi, cho bạn biết chính xác file nào và dòng nào gây lỗi.
    print("="*80 + "\n")
    print(">>> ACTION: Check the error message above (e.g., 'No module named 'langdetect'').")
    print(">>>         Then, add the missing library to the gateway's Dockerfile and rebuild the image.")

# --- Phần còn lại của file giữ nguyên ---

# --- Gateway Configuration ---
BEIT3_WORKER_URL = "http://model-workers:8001/embed"
OPENCLIP_WORKER_URL = "http://model-workers:8002/embed"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
BEIT3_COLLECTION = "beit3_large_embeddings"
OPENCLIP_COLLECTION = "openclip_h14_embeddings"
SEARCH_DEPTH = 500
TOP_K_RESULTS = 50
ALLOWED_BASE_DIR = COMMON_PARENT_DIR

# --- FastAPI App Initialization ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("--- Gateway Server: Connecting to Milvus... ---")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Gateway Server: Milvus connection successful. ---")
    except Exception as e:
        print(f"FATAL: Could not connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}. Error: {e}")

# ... (Các hàm helper khác giữ nguyên)
def search_milvus(collection_name: str, query_vector, limit: int):
    # ...
    try:
        if not utility.has_collection(collection_name): return []
        collection = Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "COSINE", "params": {"ef": max(256, limit + 50)}}
        results = collection.search(query_vector, "embedding", search_params, limit=limit, output_fields=["filepath"])
        return [{"filepath": hit.entity.get("filepath"), "score": hit.distance} for hit in results[0]] if results else []
    except Exception as e:
        print(f"ERROR during Milvus search on '{collection_name}': {e}")
        return []
def convert_distance_to_similarity(results):
    for result in results: result['score'] = max(0, 1.0 - result.get('score', 1.0))
    return results
def reciprocal_rank_fusion(results_lists: dict, weights: dict, k_rrf: int = 60):
    rrf_scores = defaultdict(float)
    raw_scores = defaultdict(dict)
    for model_name, results in results_lists.items():
        similarity_results = convert_distance_to_similarity(results)
        for rank, result in enumerate(similarity_results, 1):
            filepath = result['filepath']
            rrf_scores[filepath] += weights[model_name] * (1.0 / (k_rrf + rank))
            raw_scores[filepath][model_name] = result.get('score', 0.0)
    fused_results = [{"filepath": fp, "rrf_score": score, "beit3_sim": raw_scores[fp].get("beit3", 0.0), "openclip_sim": raw_scores[fp].get("openclip", 0.0)} for fp, score in rrf_scores.items()]
    return sorted(fused_results, key=lambda x: x['rrf_score'], reverse=True)

# ... (Các API endpoint khác giữ nguyên)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = os.path.join(ALLOWED_BASE_DIR, "Cubi", "ui", "ui1.html")
    if not os.path.exists(ui_path): raise HTTPException(status_code=500, detail=f"Error: ui1.html not found at expected path: {ui_path}")
    with open(ui_path, "r") as f: return HTMLResponse(content=f.read())
@app.post("/search")
async def search_unified(request: Request, query_text: str = Form(None), query_image: UploadFile = File(None)):
    if not query_text and not query_image: raise HTTPException(status_code=400, detail="Provide query_text or query_image")
    final_query_text_for_models = query_text
    if query_text:
        print(f"--- Original user query: '{query_text}' ---")
        translated_query = translate_text(query_text)
        if translated_query != query_text: print(f"--- Translated query for models: '{translated_query}' ---")
        final_query_text_for_models = translated_query
    files = None; data = None
    if query_image:
        image_bytes = await query_image.read()
        files = {'image_file': (query_image.filename, image_bytes, query_image.content_type)}
    if final_query_text_for_models: data = {'text_query': final_query_text_for_models}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            beit3_task = client.post(BEIT3_WORKER_URL, files=files, data=data)
            openclip_task = client.post(OPENCLIP_WORKER_URL, files=files, data=data)
            responses = await asyncio.gather(beit3_task, openclip_task, return_exceptions=True)
        except httpx.ConnectError as e: raise HTTPException(status_code=503, detail=f"Gateway could not connect to a model worker. Error: {e}")
    if isinstance(responses[0], Exception) or responses[0].status_code != 200:
        error_detail = responses[0].text if hasattr(responses[0], 'text') else str(responses[0])
        raise HTTPException(status_code=500, detail=f"BEiT-3 worker failed: {error_detail}")
    if isinstance(responses[1], Exception) or responses[1].status_code != 200:
        error_detail = responses[1].text if hasattr(responses[1], 'text') else str(responses[1])
        raise HTTPException(status_code=500, detail=f"OpenCLIP worker failed: {error_detail}")
    vec_beit3 = responses[0].json()['embedding']; vec_opc = responses[1].json()['embedding']
    beit3_results = search_milvus(BEIT3_COLLECTION, vec_beit3, SEARCH_DEPTH)
    openclip_results = search_milvus(OPENCLIP_COLLECTION, vec_opc, SEARCH_DEPTH)
    results_for_fusion = {"beit3": beit3_results, "openclip": openclip_results}
    weights = {"beit3": 0.6, "openclip": 0.4}
    final_results = reciprocal_rank_fusion(results_for_fusion, weights)[:TOP_K_RESULTS]
    base_url = str(request.base_url)
    for item in final_results:
        encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
        item['url'] = f"{base_url}images/{encoded_path}"
    return final_results
@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Invalid base64 encoding.")
    path_in_container = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo"
    remapped_path = original_path.replace("/workspace/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo", path_in_container, 1)
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(remapped_path)
    if not safe_path.startswith(safe_base): raise HTTPException(status_code=403, detail=f"Forbidden path. Safe base: {safe_base}, Requested path: {safe_path}")
    if not os.path.isfile(safe_path): raise HTTPException(status_code=404, detail=f"Image file not found at calculated path: {safe_path}")
    return FileResponse(safe_path)