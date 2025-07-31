# FILE: gateway_server2_temporal.py (Đã sửa lỗi định dạng vector cho Milvus)

import os
import sys
import traceback
import base64
import asyncio
from collections import defaultdict
import httpx
import re
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pymilvus import Collection, connections, utility
from elasticsearch import Elasticsearch

# --- Thiết lập đường dẫn ---
COMMON_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)

# --- Import các hàm xử lý truy vấn với fallback an toàn ---
try:
    from Cubi.translate_query import translate_text, enhancing, expanding
    print("--- Gateway Server: Đã import thành công 'translate_text', 'enhancing', 'expanding' từ thư mục Cubi. ---")
except ImportError:
    print("\n" + "="*80)
    print("!!! CẢNH BÁO IMPORT: Không thể import 'enhancing' hoặc 'expanding'. Tạo hàm DUMMY. !!!")
    traceback.print_exc()
    def enhancing(query: str) -> str: return f"enhanced version of {query}"
    def expanding(query: str) -> list[str]: return [query, f"variant of {query} 1", f"variant of {query} 2"]
    try:
        from Cubi.translate_query import translate_text
        print("--- Gateway Server: Đã import thành công 'translate_text'. ---")
    except ImportError:
        print("!!! LỖI NGHIÊM TRỌNG: Thiếu 'translate_text'. Ứng dụng sẽ không hoạt động đúng. !!!")
        sys.exit(1)
    print("="*80 + "\n")


# --- Cấu hình ---
BEIT3_WORKER_URL = "http://model-workers:8001/embed"
OPENCLIP_WORKER_URL = "http://model-workers:8002/embed"
ELASTICSEARCH_HOST = "http://elasticsearch2:9200"
OCR_INDEX_NAME = "opencubee_2" 
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"

BEIT3_COLLECTION = "beit3_video_frame_embeddings"
OPENCLIP_COLLECTION = "openclip_h14_video_embeddings"

SEARCH_DEPTH = 500
TOP_K_RESULTS = 100
ALLOWED_BASE_DIR = COMMON_PARENT_DIR
MAX_SEQUENCES_TO_RETURN = 50
INITIAL_CANDIDATES = 50
SEARCH_DEPTH_PER_STAGE = 200

es = None

# --- Khởi tạo FastAPI & Sự kiện Startup ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global es
    print("--- Gateway Server: Đang kết nối tới Milvus... ---")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Gateway Server: Kết nối Milvus thành công. ---")
    except Exception as e:
        print(f"LỖI: Không thể kết nối tới Milvus tại {MILVUS_HOST}:{MILVUS_PORT}. Lỗi: {e}")
    
    print("--- Gateway Server: Đang kết nối tới Elasticsearch... ---")
    try:
        es = Elasticsearch(ELASTICSEARCH_HOST)
        if not es.ping(): raise ConnectionError("Ping Elasticsearch thất bại.")
        print("--- Gateway Server: Kết nối Elasticsearch thành công. ---")
    except Exception as e:
        print(f"LỖI: Không thể kết nối tới Elasticsearch tại {ELASTICSEARCH_HOST}. Lỗi: {e}")
        es = None

# --- Pydantic Models cho các Request từ API ---
class StageData(BaseModel):
    query: str
    expand: bool
    enhance: bool

class TemporalSearchRequest(BaseModel):
    stages: list[StageData]

class OcrSearchRequest(BaseModel):
    query: str
    expand: bool
    enhance: bool

# --- Các hàm hỗ trợ ---

def search_milvus(collection_name: str, query_vectors: list, limit: int, min_filepath: str = None):
    try:
        if not utility.has_collection(collection_name) or not len(query_vectors):
            return []
        
        collection = Collection(collection_name)
        collection.load()
        expr = f"filepath > '{min_filepath}'" if min_filepath else None
        
        search_params = {"metric_type": "COSINE", "params": {"ef": max(256, limit + 50)}}
        output_fields = ["filepath", "video_id", "frame_id", "id"]
        
        # Dòng này là nơi lỗi xảy ra. Giờ nó sẽ nhận được dữ liệu đúng định dạng.
        results = collection.search(query_vectors, "embedding", search_params, limit=limit, output_fields=output_fields, expr=expr)
        
        all_hits = []
        for result_set in results:
            for hit in result_set:
                all_hits.append({
                    "filepath": hit.entity.get("filepath"), 
                    "score": hit.distance,
                    "video_id": hit.entity.get("video_id"),
                    "frame_id": hit.entity.get("frame_id"),
                    "id": hit.entity.get("id"),
                })
        return all_hits
    except Exception as e:
        print(f"Lỗi trong quá trình tìm kiếm Milvus trên '{collection_name}': {e}")
        traceback.print_exc()
        return []

def convert_distance_to_similarity(results):
    for result in results: result['score'] = max(0, 1.0 - result.get('score', 1.0))
    return results

def reciprocal_rank_fusion(results_lists: dict, weights: dict, k_rrf: int = 60):
    rrf_scores, raw_scores, filepath_data = defaultdict(float), defaultdict(dict), {}
    for model_name, results in results_lists.items():
        similarity_results = convert_distance_to_similarity(results)
        for rank, result in enumerate(similarity_results, 1):
            filepath = result.get('filepath')
            if not filepath: continue
            model_weight = weights.get(model_name, 1)
            rrf_scores[filepath] += model_weight * (1.0 / (k_rrf + rank))
            raw_scores[filepath][model_name] = result.get('score', 0.0)
            if filepath not in filepath_data: filepath_data[filepath] = result
    fused_results = []
    for fp, score in rrf_scores.items():
        base_data = filepath_data[fp]
        base_data['rrf_score'] = score
        base_data['beit3_sim'] = raw_scores[fp].get("beit3", 0.0)
        base_data['openclip_sim'] = raw_scores[fp].get("openclip", 0.0)
        base_data.pop('score', None)
        fused_results.append(base_data)
    return sorted(fused_results, key=lambda x: x['rrf_score'], reverse=True)

def search_ocr_on_elasticsearch(keyword: str, limit: int=100):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text", "asr_text"]}}}
    try: response = es.search(index=OCR_INDEX_NAME, body=query, size=limit)
    except Exception as e: print(f"Lỗi Elasticsearch: {e}"); return [] 
    results = []
    base_image_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/Enn/dataset/data_frame_ocr_png"
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        if not all([source.get('video'), source.get('shot_id'), source.get('frame')]): continue
        filename = f"{source['video']}_{source['shot_id']}_{str(source['frame']).zfill(6)}.png"
        results.append({
            "filepath": os.path.join(base_image_path, filename), "score": hit['_score'],
            "ocr_text": source.get('ocr_text', 'N/A'), "video": source.get('video'), "shot_id": source.get('shot_id'),
        })
    return results

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = os.path.join(ALLOWED_BASE_DIR, "Cubi", "ui", "ui1_2_temporal.html")
    if not os.path.exists(ui_path): raise HTTPException(status_code=500, detail=f"Lỗi: không tìm thấy file ui1_2_temporal.html tại: {ui_path}")
    with open(ui_path, "r") as f: return HTMLResponse(content=f.read())

async def get_stage_results(client: httpx.AsyncClient, stage: StageData, min_filepath: str = None) -> list[dict]:
    base_query = translate_text(stage.query)
    print(f"  - Query gốc: '{stage.query}' -> Đã dịch: '{base_query}'")

    queries_to_process = [base_query]
    if stage.expand:
        queries_to_process = expanding(base_query)
        print(f"  - Đã mở rộng thành {len(queries_to_process)} truy vấn: {queries_to_process}")

    if stage.enhance:
        enhanced_queries = [enhancing(q) for q in queries_to_process]
        print(f"  - Đã cải thiện {len(enhanced_queries)} truy vấn: {enhanced_queries}")
        queries_to_embed = enhanced_queries
    else:
        queries_to_embed = queries_to_process

    beit3_tasks = [client.post(BEIT3_WORKER_URL, data={'text_query': q}) for q in queries_to_embed]
    openclip_tasks = [client.post(OPENCLIP_WORKER_URL, data={'text_query': q}) for q in queries_to_embed]
    
    all_responses = await asyncio.gather(*beit3_tasks, *openclip_tasks)
    
    num_queries = len(queries_to_embed)
    
    # === FIX HERE: Thêm [0] để loại bỏ lớp list thừa ===
    beit3_vectors_list = [r.json()['embedding'][0] for r in all_responses[:num_queries]]
    openclip_vectors_list = [r.json()['embedding'][0] for r in all_responses[num_queries:]]

    np_beit3_vectors = np.array(beit3_vectors_list, dtype=np.float32)
    np_openclip_vectors = np.array(openclip_vectors_list, dtype=np.float32)

    beit3_results = search_milvus(BEIT3_COLLECTION, np_beit3_vectors, SEARCH_DEPTH_PER_STAGE, min_filepath=min_filepath)
    openclip_results = search_milvus(OPENCLIP_COLLECTION, np_openclip_vectors, SEARCH_DEPTH_PER_STAGE, min_filepath=min_filepath)

    if not beit3_results and not openclip_results: return []

    model_weights = {"beit3": 1, "openclip": 1}
    return reciprocal_rank_fusion({"beit3": beit3_results, "openclip": openclip_results}, model_weights)

@app.post("/temporal_search")
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    stages = request_data.stages
    if not stages: raise HTTPException(status_code=400, detail="Không có stage nào được cung cấp.")
    print("\n" + "="*50 + "\n--- BẮT ĐẦU TÌM KIẾM TEMPORAL ĐA CHUỖI ---\n" + "="*50)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"\n--- Xử lý Stage 1: '{stages[0].query}' (Enhance: {stages[0].enhance}, Expand: {stages[0].expand}) để tìm điểm bắt đầu ---")
        initial_candidates = (await get_stage_results(client, stages[0]))[:INITIAL_CANDIDATES]

        if not initial_candidates:
            print("--- !!! TÌM KIẾM THẤT BẠI: Không tìm thấy ứng viên nào cho stage đầu tiên. !!! ---")
            return []
        print(f"--- Đã tìm thấy {len(initial_candidates)} shot có thể bắt đầu. ---")

        all_valid_sequences = []
        for candidate in initial_candidates:
            video_id = candidate.get('video_id')
            if not video_id: continue

            print(f"\n  -> Phân nhánh từ ứng viên: '{os.path.basename(candidate['filepath'])}' (Video: {video_id})")
            current_sequence, last_successful_path, is_complete = [candidate], candidate['filepath'], True

            for i, stage in enumerate(stages[1:]):
                stage_num = i + 2
                print(f"    -- Đang tìm Stage {stage_num}: '{stage.query}' (Enhance: {stage.enhance}, Expand: {stage.expand}) trong Video {video_id} --")
                stage_results = await get_stage_results(client, stage, min_filepath=last_successful_path)
                filtered_by_video = [res for res in stage_results if res.get('video_id') == video_id]
                
                if not filtered_by_video:
                    print(f"    -- !!! Chuỗi bị đứt. Không tìm thấy kết quả cho Stage {stage_num} trong Video {video_id}. !!! --")
                    is_complete = False; break
                
                top_result = filtered_by_video[0]
                current_sequence.append(top_result)
                last_successful_path = top_result['filepath']
                print(f"    -- Đã tìm thấy kết quả cho Stage {stage_num}: '{os.path.basename(last_successful_path)}' --")

            if is_complete:
                print(f"  -> THÀNH CÔNG: Tìm thấy một chuỗi hoàn chỉnh gồm {len(current_sequence)} shot cho Video {video_id}.")
                all_valid_sequences.append(current_sequence)
    
    print(f"\n--- TÌM KIẾM HOÀN TẤT: Tìm thấy {len(all_valid_sequences)} chuỗi hợp lệ. ---")
    
    processed_sequences = []
    for seq in all_valid_sequences:
        if not seq: continue
        processed_sequences.append({
            "video_id": seq[0].get('video_id'),
            "average_score": sum(shot['rrf_score'] for shot in seq) / len(seq),
            "shots": seq
        })
    processed_sequences.sort(key=lambda x: x['average_score'], reverse=True)

    final_sequences_data = processed_sequences[:MAX_SEQUENCES_TO_RETURN]
    base_url = str(request.base_url)
    for seq_data in final_sequences_data:
        for item in seq_data['shots']:
            encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
            item['url'] = f"{base_url}images/{encoded_path}"
    return final_sequences_data

@app.post("/ocr_search")
async def ocr_search(request_data: OcrSearchRequest, request: Request):
    if not request_data.query: raise HTTPException(status_code=400, detail="Không có từ khóa nào được cung cấp.")

    base_query = translate_text(request_data.query)
    print(f"Tìm kiếm OCR - Gốc: '{request_data.query}' -> Đã dịch: '{base_query}' (Enhance: {request_data.enhance}, Expand: {request_data.expand})")

    queries_to_process = [base_query]
    if request_data.expand:
        queries_to_process = expanding(base_query)
        print(f"  - Đã mở rộng thành: {queries_to_process}")

    if request_data.enhance:
        enhanced_queries = [enhancing(q) for q in queries_to_process]
        print(f"  - Đã cải thiện thành: {enhanced_queries}")
        queries_to_search = enhanced_queries
    else:
        queries_to_search = queries_to_process

    all_results, seen_filepaths = [], set()
    for keyword in queries_to_search:
        results = search_ocr_on_elasticsearch(keyword, limit=TOP_K_RESULTS)
        for res in results:
            if res['filepath'] not in seen_filepaths:
                all_results.append(res); seen_filepaths.add(res['filepath'])

    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:TOP_K_RESULTS]
    if not sorted_results: return []

    base_url = str(request.base_url)
    for item in sorted_results:
        encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
        item['url'] = f"{base_url}images/{encoded_path}"
    return sorted_results

@app.post("/search")
async def search_unified(request: Request, query_text: str = Form(None), query_image: UploadFile = File(None)):
    if not query_text and not query_image: raise HTTPException(status_code=400, detail="Cung cấp văn bản hoặc hình ảnh.")
    final_query_text_for_models = query_text
    if query_text: final_query_text_for_models = translate_text(query_text)
    files, data = None, None
    if query_image: files = {'image_file': (query_image.filename, await query_image.read(), query_image.content_type)}
    if final_query_text_for_models: data = {'text_query': final_query_text_for_models}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            responses = await asyncio.gather(
                client.post(BEIT3_WORKER_URL, files=files, data=data),
                client.post(OPENCLIP_WORKER_URL, files=files, data=data), return_exceptions=True
            )
        except httpx.ConnectError as e: raise HTTPException(status_code=503, detail=f"Gateway không thể kết nối đến model worker. Lỗi: {e}")

    for i, resp in enumerate(responses):
        if isinstance(resp, Exception) or resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Model worker {i+1} thất bại.")
    
    # === FIX HERE: Sửa cách tạo np.array để có shape 2D đúng ===
    vec_beit3 = np.array(responses[0].json()['embedding'], dtype=np.float32)
    vec_opc = np.array(responses[1].json()['embedding'], dtype=np.float32)

    beit3_results = search_milvus(BEIT3_COLLECTION, vec_beit3, SEARCH_DEPTH)
    openclip_results = search_milvus(OPENCLIP_COLLECTION, vec_opc, SEARCH_DEPTH)
    
    model_weights = {"beit3": 1, "openclip": 1}
    final_results = reciprocal_rank_fusion({"beit3": beit3_results, "openclip": openclip_results}, model_weights)[:TOP_K_RESULTS]
    
    base_url = str(request.base_url)
    for item in final_results:
        encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
        item['url'] = f"{base_url}images/{encoded_path}"
    return final_results

@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Mã base64 không hợp lệ.")
    remapped_path = original_path.replace("/workspace/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo", "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo", 1)
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(remapped_path)
    if not safe_path.startswith(safe_base): raise HTTPException(status_code=403, detail="Đường dẫn bị cấm.")
    if not os.path.isfile(safe_path): raise HTTPException(status_code=404, detail=f"Không tìm thấy file ảnh tại: {safe_path}")
    return FileResponse(safe_path)