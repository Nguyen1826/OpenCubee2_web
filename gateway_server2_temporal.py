# FILE: gateway_server.py (Phiên bản Gốc được Sửa lỗi và Nâng cấp)

import os
import sys
import traceback
import base64
import asyncio
from collections import defaultdict
import httpx
import numpy as np
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pymilvus import Collection, connections, utility
from elasticsearch import Elasticsearch

# --- Thiết lập đường dẫn ---
_CURRENT_DIR_PARENT = os.path.dirname(os.path.abspath(__file__))
COMMON_PARENT_DIR = os.path.dirname(_CURRENT_DIR_PARENT)
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)
# ALLOWED_BASE_DIR trỏ đến thư mục gốc của dự án để truy cập được cả Repo và dataset_test
ALLOWED_BASE_DIR = "/app"

# --- Import các hàm xử lý truy vấn ---
try:
    from translate_query import translate_text, enhancing, expanding
    print("--- Gateway Server: Đã import thành công các hàm xử lý truy vấn. ---")
except ImportError:
    print("!!! CẢNH BÁO: Không thể import các hàm xử lý truy vấn. Sử dụng hàm DUMMY. !!!")
    def enhancing(q: str) -> str: return q
    def expanding(q: str) -> list[str]: return [q]
    def translate_text(q: str) -> str: return q

# --- Cấu hình ---
BEIT3_WORKER_URL = "http://model-workers:8001/embed"; OPENCLIP_WORKER_URL = "http://model-workers:8002/embed"
ELASTICSEARCH_HOST = "http://elasticsearch2:9200"; OCR_INDEX_NAME = "opencubee_2"
MILVUS_HOST = "milvus-standalone"; MILVUS_PORT = "19530"
BEIT3_COLLECTION = "beit3_video_frame_embeddings"; OPENCLIP_COLLECTION = "openclip_h14_video_embeddings"

# Sử dụng trọng số và ngưỡng từ script gốc của bạn
MODEL_WEIGHTS = {"beit3": 0.6, "openclip": 0.4}
SIMILARITY_THRESHOLD = 0.25
SEARCH_DEPTH = 500
TOP_K_RESULTS = 50
MAX_SEQUENCES_TO_RETURN = 20
INITIAL_CANDIDATES = 20
SEARCH_DEPTH_PER_STAGE = 200

es = None; app = FastAPI()

@app.on_event("startup")
def startup_event():
    global es
    print("--- Gateway Server: Connecting to services... ---")
    try: connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT); print("--- Milvus connection successful. ---")
    except Exception as e: print(f"FATAL: Could not connect to Milvus. Error: {e}")
    try: es = Elasticsearch(ELASTICSEARCH_HOST); es.ping(); print("--- Elasticsearch connection successful. ---")
    except Exception as e: print(f"FATAL: Could not connect to Elasticsearch. Error: {e}"); es = None

# --- Pydantic Models ---
class StageData(BaseModel): query: str; expand: bool; enhance: bool
class TemporalSearchRequest(BaseModel): stages: list[StageData]; models: List[str] = ["beit3", "openclip"]
class OcrSearchRequest(BaseModel): query: str; expand: bool; enhance: bool

# --- Các hàm hỗ trợ ---
def search_milvus(collection_name: str, query_vectors: list, limit: int, min_filepath: str = None):
    try:
        if not utility.has_collection(collection_name) or not len(query_vectors): return []
        collection = Collection(collection_name); collection.load()
        expr = f"filepath > '{min_filepath}'" if min_filepath else None
        # Sửa tham số tìm kiếm cho SCANN
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 128, "reorder_k": max(limit, 200)}}
        output_fields = ["filepath", "video_id", "frame_id"]
        results = collection.search(query_vectors, "embedding", search_params, limit=limit, output_fields=output_fields, expr=expr)
        return [{"filepath": h.entity.get("filepath"), "score": h.distance, "video_id": h.entity.get("video_id"), "frame_id": h.entity.get("frame_id")} for s in results for h in s]
    except Exception as e: print(f"ERROR during Milvus search on '{collection_name}': {e}"); return []

def convert_distance_to_similarity(results):
    for result in results: result['score'] = max(0, 1.0 - result.get('score', 1.0))
    return results

def reciprocal_rank_fusion(results_lists: dict, weights: dict, k_rrf: int = 60):
    master_data = defaultdict(lambda: {"raw_scores": {}})
    for model_name, results in results_lists.items():
        if not results: continue
        similarity_results = convert_distance_to_similarity(results)
        for rank, result in enumerate(similarity_results, 1):
            filepath = result.get('filepath')
            if not filepath: continue
            if 'metadata' not in master_data[filepath]: master_data[filepath]['metadata'] = result
            master_data[filepath]['raw_scores'][model_name] = { "score": result.get('score', 0.0), "rank": rank }
    if not master_data: return []
    final_results = []
    for filepath, data in master_data.items():
        rrf_score = 0.0
        for model_name, score_info in data['raw_scores'].items():
            model_weight = weights.get(model_name, 1.0); rank = score_info['rank']
            rrf_score += model_weight * (1.0 / (k_rrf + rank))
        final_item = data['metadata']; final_item['rrf_score'] = rrf_score
        final_item['beit3_sim'] = data['raw_scores'].get('beit3', {}).get('score', 0.0)
        final_item['openclip_sim'] = data['raw_scores'].get('openclip', {}).get('score', 0.0)
        final_item.pop('score', None); final_results.append(final_item)
    if final_results:
        max_rrf_score = max(item['rrf_score'] for item in final_results)
        if max_rrf_score > 0:
            for item in final_results: item['rrf_score'] /= max_rrf_score
    return sorted(final_results, key=lambda x: x['rrf_score'], reverse=True)

def search_ocr_on_elasticsearch(keyword: str, limit: int=100):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text", "asr_text"]}}}
    try: response = es.search(index=OCR_INDEX_NAME, body=query, size=limit)
    except Exception as e: print(f"Lỗi Elasticsearch: {e}"); return []
    results = []
    # Giữ nguyên đường dẫn cũ cho dữ liệu OCR
    base_image_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/Enn/dataset/data_frame_ocr_png"
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        if not all([source.get('video'), source.get('shot_id'), source.get('frame')]): continue
        filename = f"{source['video']}_{source['shot_id']}_{str(source['frame']).zfill(6)}.png"
        results.append({"filepath": os.path.join(base_image_path, filename), "score": hit['_score'], "ocr_text": source.get('ocr_text', 'N/A'), "video": source.get('video'), "shot_id": source.get('shot_id')})
    return results

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Sửa lại đường dẫn UI cho đúng
    ui_path = "./ui1_2_temporal.html"
    if not os.path.exists(ui_path): raise HTTPException(status_code=500, detail=f"Lỗi: không tìm thấy file ui1_2_temporal.html tại: {ui_path}")
    with open(ui_path, "r") as f: return HTMLResponse(content=f.read())

@app.post("/search")
async def search_unified(request: Request, query_text: str = Form(None), query_image: UploadFile = File(None), models: List[str] = Form(...)):
    print(f"\n--- BẮT ĐẦU TÌM KIẾM ĐƠN GIẢN (UNIFIED) - Models: {models} ---")
    if not query_text and not query_image: raise HTTPException(status_code=400, detail="Cung cấp văn bản hoặc hình ảnh.")
    if not models: raise HTTPException(status_code=400, detail="Phải chọn ít nhất một model.")
    final_query_text_for_models = translate_text(query_text) if query_text else None
    files, data = None, None
    if query_image: files = {'image_file': (query_image.filename, await query_image.read(), query_image.content_type)}
    if final_query_text_for_models: data = {'text_query': final_query_text_for_models}
    
    vec_beit3, vec_opc = [], []
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        if "beit3" in models: tasks.append(client.post(BEIT3_WORKER_URL, files=files, data=data))
        if "openclip" in models: tasks.append(client.post(OPENCLIP_WORKER_URL, files=files, data=data))
        try: responses = await asyncio.gather(*tasks, return_exceptions=True)
        except httpx.ConnectError as e: raise HTTPException(status_code=503, detail=f"Lỗi kết nối worker: {e}")
    
    response_iter = iter(responses)
    if "beit3" in models:
        resp = next(response_iter);
        if isinstance(resp, Exception) or resp.status_code != 200: raise HTTPException(status_code=500, detail="BEiT-3 worker thất bại.")
        vec_beit3 = np.array(resp.json()['embedding'], dtype=np.float32).tolist()
    if "openclip" in models:
        resp = next(response_iter);
        if isinstance(resp, Exception) or resp.status_code != 200: raise HTTPException(status_code=500, detail="OpenCLIP worker thất bại.")
        vec_opc = np.array(resp.json()['embedding'], dtype=np.float32).tolist()

    beit3_results = search_milvus(BEIT3_COLLECTION, vec_beit3, SEARCH_DEPTH) if vec_beit3 else []
    openclip_results = search_milvus(OPENCLIP_COLLECTION, vec_opc, SEARCH_DEPTH) if vec_opc else []
    
    fused_results = reciprocal_rank_fusion({"beit3": beit3_results, "openclip": openclip_results}, MODEL_WEIGHTS)
    
    # Áp dụng cổng chất lượng từ script gốc
    quality_filtered_results = [r for r in fused_results if ((r.get('beit3_sim',0)+r.get('openclip_sim',0))/2 if (r.get('beit3_sim') and r.get('openclip_sim')) else max(r.get('beit3_sim',0), r.get('openclip_sim',0))) >= SIMILARITY_THRESHOLD]
    
    final_results = quality_filtered_results[:TOP_K_RESULTS]
    processed_sequences = [{"video_id": shot.get('video_id'), "shots": [shot]} for shot in final_results]
    
    base_url = str(request.base_url)
    for seq_data in processed_sequences:
        for item in seq_data['shots']: item['url'] = f"{base_url}images/{base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')}"
    return processed_sequences

@app.post("/temporal_search")
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    models_to_use = request_data.models
    print(f"\n--- BẮT ĐẦU TÌM KIẾM NÂNG CAO/TEMPORAL - Models: {models_to_use} ---")
    stages = request_data.stages
    if not stages: raise HTTPException(status_code=400, detail="Không có stage nào được cung cấp.")
    if not models_to_use: raise HTTPException(status_code=400, detail="Phải chọn ít nhất một model.")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async def get_stage_results(stage: StageData, min_filepath: str = None) -> list[dict]:
            base_query = translate_text(stage.query); queries_to_process = expanding(base_query) if stage.expand else [base_query]
            queries_to_embed = [enhancing(q) for q in queries_to_process] if stage.enhance else queries_to_process
            tasks = []
            if "beit3" in models_to_use: tasks.extend([client.post(BEIT3_WORKER_URL, data={'text_query': q}) for q in queries_to_embed])
            if "openclip" in models_to_use: tasks.extend([client.post(OPENCLIP_WORKER_URL, data={'text_query': q}) for q in queries_to_embed])
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            beit3_vecs, openclip_vecs = [], []; response_iter = iter(all_responses); num_queries = len(queries_to_embed)
            if "beit3" in models_to_use: beit3_vecs = [next(response_iter).json()['embedding'][0] for _ in range(num_queries)]
            if "openclip" in models_to_use: openclip_vecs = [next(response_iter).json()['embedding'][0] for _ in range(num_queries)]
            beit3_res = search_milvus(BEIT3_COLLECTION, beit3_vecs, SEARCH_DEPTH_PER_STAGE, min_filepath) if beit3_vecs else []
            openclip_res = search_milvus(OPENCLIP_COLLECTION, openclip_vecs, SEARCH_DEPTH_PER_STAGE, min_filepath) if openclip_vecs else []
            fused_res = reciprocal_rank_fusion({"beit3": beit3_res, "openclip": openclip_res}, MODEL_WEIGHTS)
            return [r for r in fused_res if ((r.get('beit3_sim',0)+r.get('openclip_sim',0))/2 if (r.get('beit3_sim') and r.get('openclip_sim')) else max(r.get('beit3_sim',0), r.get('openclip_sim',0))) >= SIMILARITY_THRESHOLD]

        initial_candidates = (await get_stage_results(stages[0]))[:INITIAL_CANDIDATES]
        if not initial_candidates: return []
        all_valid_sequences = []
        for candidate in initial_candidates:
            video_id = candidate.get('video_id')
            if not video_id: continue
            current_sequence, last_path, is_complete = [candidate], candidate['filepath'], True
            for stage in stages[1:]:
                stage_res = await get_stage_results(stage, min_filepath=last_path)
                filtered = [r for r in stage_res if r.get('video_id') == video_id]
                if not filtered: is_complete = False; break
                top_result = filtered[0]; current_sequence.append(top_result); last_path = top_result['filepath']
            if is_complete: all_valid_sequences.append(current_sequence)
        
        processed_sequences = []
        for seq in all_valid_sequences:
            if not seq: continue
            avg_rrf = sum(shot.get('rrf_score', 0) for shot in seq) / len(seq) if seq else 0
            processed_sequences.append({"video_id": seq[0].get('video_id'), "average_rrf_score": avg_rrf, "shots": seq})
        processed_sequences.sort(key=lambda x: x['average_rrf_score'], reverse=True)

    final_sequences_data = processed_sequences[:MAX_SEQUENCES_TO_RETURN]
    base_url = str(request.base_url)
    for seq_data in final_sequences_data:
        for item in seq_data['shots']: item['url'] = f"{base_url}images/{base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')}"
    return final_sequences_data

@app.post("/ocr_search")
async def ocr_search(request_data: OcrSearchRequest, request: Request):
    print("\n--- BẮT ĐẦU TÌM KIẾM OCR ---")
    if not request_data.query: raise HTTPException(status_code=400, detail="Không có từ khóa nào được cung cấp.")
    base_query = translate_text(request_data.query); queries_to_process = expanding(base_query) if request_data.expand else [base_query]
    queries_to_search = [enhancing(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    all_results, seen_filepaths = [], set()
    for keyword in queries_to_search:
        results = search_ocr_on_elasticsearch(keyword, limit=TOP_K_RESULTS)
        for res in results:
            if res['filepath'] not in seen_filepaths: all_results.append(res); seen_filepaths.add(res['filepath'])
    sorted_results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:TOP_K_RESULTS]
    if not sorted_results: return []
    base_url = str(request.base_url)
    for item in sorted_results: item['url'] = f"{base_url}images/{base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')}"
    return sorted_results
    
@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Mã base64 không hợp lệ.")
    # Sửa remapping path cho linh hoạt
    if original_path.startswith("/workspace"): remapped_path = original_path.replace("/workspace", "/app", 1)
    else: remapped_path = original_path
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    try: safe_path = os.path.realpath(remapped_path)
    except FileNotFoundError: raise HTTPException(status_code=404, detail=f"Không tìm thấy file tại đường dẫn đã ánh xạ: {remapped_path}")
    if not safe_path.startswith(safe_base): raise HTTPException(status_code=403, detail="Đường dẫn bị cấm.")
    if not os.path.isfile(safe_path): raise HTTPException(status_code=404, detail=f"Không tìm thấy file ảnh tại: {safe_path}")
    return FileResponse(safe_path)