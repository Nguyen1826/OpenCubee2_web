# FILE: gateway_server.py (FINAL - Correct Temporal Clustering Logic - Full Cluster Return)

import os
import sys
import time
import traceback
import base64
import asyncio
from collections import defaultdict
import httpx
from typing import List, Dict, Any
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
BEIT3_WORKER_URL = "http://model-workers:8001/embed"
BGE_WORKER_URL = "http://model-workers:8002/embed"
ELASTICSEARCH_HOST = "http://elasticsearch2:9200"; 
OCR_ASR_INDEX_NAME = "opencubee_2"
MILVUS_HOST = "milvus-standalone"; MILVUS_PORT = "19530"
BEIT3_COLLECTION = "beit3_image_caption_embeddings"
BGE_COLLECTION = "bge_vl_large_image_embeddings" 
MODEL_WEIGHTS = {"beit3": 0.6, "bge": 0.4}; 
SEARCH_DEPTH = 1000; TOP_K_RESULTS = 50; MAX_SEQUENCES_TO_RETURN = 20; SEARCH_DEPTH_PER_STAGE = 200

es = None; app = FastAPI()

# --- Pydantic Models ---
class StageDataBase(BaseModel): query: str; expand: bool; enhance: bool
class TemporalSearchRequest(BaseModel): stages: list[StageDataBase]; models: List[str] = ["beit3", "bge"]; cluster: bool = False
class OcrSearchRequest(BaseModel): query: str; expand: bool; enhance: bool
class AsrSearchRequest(BaseModel): query: str; expand: bool; enhance: bool
class ProcessQueryRequest(BaseModel): query: str; enhance: bool; expand: bool

@app.on_event("startup")
def startup_event():
    global es
    try: connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT); print("--- Milvus connection successful. ---")
    except Exception as e: print(f"FATAL: Could not connect to Milvus. Error: {e}")
    try: es = Elasticsearch(ELASTICSEARCH_HOST); es.ping(); print("--- Elasticsearch connection successful. ---")
    except Exception as e: print(f"FATAL: Could not connect to Elasticsearch. Error: {e}"); es = None

# --- Các hàm hỗ trợ ---
def search_milvus(collection_name: str, query_vectors: list, limit: int, expr: str = None):
    try:
        if not utility.has_collection(collection_name) or not len(query_vectors): return []
        collection = Collection(collection_name); collection.load()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 128}}
        output_fields = ["filepath", "video_id", "shot_id", "frame_id"]
        results = collection.search(query_vectors, "image_embedding", search_params, limit=limit, output_fields=output_fields, expr=expr)
        parsed_results = []
        for s in results:
            for h in s:
                res = {
                    "filepath": h.entity.get("filepath"), "score": h.distance, "video_id": h.entity.get("video_id"),
                    "frame_id": h.entity.get("frame_id"), "shot_id": str(h.entity.get("shot_id"))
                }
                parsed_results.append(res)
        return parsed_results
    except Exception as e: 
        print(f"ERROR during Milvus search on '{collection_name}': {e}"); traceback.print_exc(); return []

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
            rrf_score += weights.get(model_name, 1.0) * (1.0 / (k_rrf + score_info['rank']))
        final_item = data['metadata']; final_item['rrf_score'] = rrf_score
        final_item.pop('score', None); final_results.append(final_item)
    return sorted(final_results, key=lambda x: x['rrf_score'], reverse=True)

def search_ocr_on_elasticsearch(keyword: str, limit: int=100):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text"]}}}
    try: response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
    except Exception as e: print(f"Lỗi Elasticsearch: {e}"); return []
    results = []
    base_image_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/Enn/dataset/data_frame_ocr_png"
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        if not all([source.get('video'), source.get('shot_id'), source.get('frame')]): continue
        filename = f"{source['video']}_{source['shot_id']}_{str(source['frame']).zfill(6)}.png"
        results.append({"filepath": os.path.join(base_image_path, filename), "score": hit['_score'], "video_id": source.get('video'), "shot_id": source.get('shot_id')})
    return results

def search_asr_on_elasticsearch(keyword: str, limit: int=100):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["asr_text"]}}}
    try: response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
    except Exception as e: print(f"Lỗi Elasticsearch ASR: {e}"); return []
    results = []
    base_image_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/Enn/dataset/data_frame_ocr_png"
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        if not all([source.get('video'), source.get('shot_id'), source.get('frame')]): continue
        filename = f"{source['video']}_{source['shot_id']}_{str(source['frame']).zfill(6)}.png"
        results.append({"filepath": os.path.join(base_image_path, filename), "score": hit['_score'], "video_id": source.get('video'), "shot_id": source.get('shot_id')})
    return results

def process_and_cluster_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results: return []
    shots_by_video = defaultdict(list)
    for res in results:
        if not all(k in res for k in ['video_id', 'shot_id']): continue
        try: 
            res['shot_id_int'] = int(str(res['shot_id']))
            shots_by_video[res['video_id']].append(res)
        except (ValueError, TypeError): continue
    all_clusters = []
    for video_id, shots in shots_by_video.items():
        if not shots: continue
        shots_by_shot_id = defaultdict(list); [shots_by_shot_id[shot['shot_id_int']].append(shot) for shot in shots]
        sorted_shot_ids = sorted(shots_by_shot_id.keys())
        if not sorted_shot_ids: continue
        current_cluster = []
        for i, shot_id in enumerate(sorted_shot_ids):
            if i > 0 and shot_id != sorted_shot_ids[i-1] + 1:
                if current_cluster: all_clusters.append(current_cluster)
                current_cluster = []
            current_cluster.extend(shots_by_shot_id[shot_id])
        if current_cluster: all_clusters.append(current_cluster)
    if not all_clusters: return []
    processed_clusters = []
    for cluster_shots in all_clusters:
        if not cluster_shots: continue
        sorted_cluster_shots = sorted(cluster_shots, key=lambda x: x.get('rrf_score', x.get('score', 0)), reverse=True)
        best_shot = sorted_cluster_shots[0]
        max_score = best_shot.get('rrf_score', best_shot.get('score', 0))
        processed_clusters.append({"cluster_score": max_score, "shots": sorted_cluster_shots, "best_shot": best_shot})
    return sorted(processed_clusters, key=lambda x: x['cluster_score'], reverse=True)

def add_image_urls(data: List[Dict[str, Any]], base_url: str):
    for item in data:
        for shot in item.get('shots', []):
            if shot.get('filepath') and 'url' not in shot:
                shot['url'] = f"{base_url}images/{base64.urlsafe_b64encode(shot['filepath'].encode('utf-8')).decode('utf-8')}"
        if (best_shot := item.get('best_shot')) and best_shot.get('filepath') and 'url' not in best_shot:
            best_shot['url'] = f"{base_url}images/{base64.urlsafe_b64encode(best_shot['filepath'].encode('utf-8')).decode('utf-8')}"
    return data

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = "./ui1_2_temporal.html"
    if not os.path.exists(ui_path): raise HTTPException(status_code=500, detail="UI file not found")
    with open(ui_path, "r") as f: return HTMLResponse(content=f.read())

@app.post("/process_query")
async def process_query(request_data: ProcessQueryRequest):
    if not request_data.query: return {"processed_query": ""}
    base_query = translate_text(request_data.query)
    queries_to_process = expanding(base_query) if request_data.expand else [base_query]
    final_queries = [enhancing(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    return {"processed_query": " ".join(final_queries)}

@app.post("/search")
async def search_unified(request: Request, query_text: str = Form(None), query_image: UploadFile = File(None), models: List[str] = Form(...), enhance: bool = Form(False), expand: bool = Form(False)):
    if not query_text and not query_image: raise HTTPException(status_code=400, detail="Text or image query required.")
    if not models: raise HTTPException(status_code=400, detail="At least one model must be selected.")
    print(f"Simple Search | Enhance: {enhance}, Expand: {expand}, Models: {models}")
    final_queries_to_embed = []
    if query_text:
        base_query = translate_text(query_text)
        queries_to_process = expanding(base_query) if expand else [base_query]
        final_queries_to_embed = [enhancing(q) for q in queries_to_process] if enhance else queries_to_process
    
    image_content = await query_image.read() if query_image else None
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks, model_order = [], []
        for model in models:
            url = BEIT3_WORKER_URL if model == "beit3" else (BGE_WORKER_URL if model == "bge" else None)
            if not url: continue
            if model == "bge":
                if image_content:
                    tasks.append(client.post(url, files={'image_file': (query_image.filename, image_content, query_image.content_type)}))
                    model_order.append(model)
                elif final_queries_to_embed:
                    for q in final_queries_to_embed:
                        tasks.append(client.post(url, data={'text_query': q}))
                        model_order.append(model)
            else: 
                files = {'image_file': (query_image.filename, image_content, query_image.content_type)} if image_content else None
                if final_queries_to_embed:
                    for q in final_queries_to_embed:
                        tasks.append(client.post(url, files=files, data={'text_query': q}))
                        model_order.append(model)
                elif files:
                    tasks.append(client.post(url, files=files))
                    model_order.append(model)
        if not tasks: raise HTTPException(status_code=400, detail="No valid query to process.")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results_by_model = defaultdict(list)
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception): print(f"Error calling model {model_order[i]}: {resp}"); continue
            if resp.status_code == 200: results_by_model[model_order[i]].extend(resp.json()['embedding'])
            else: print(f"Error from model {model_order[i]} (status {resp.status_code}): {resp.text}")
    beit3_res = search_milvus(BEIT3_COLLECTION, results_by_model.get("beit3", []), SEARCH_DEPTH)
    bge_res = search_milvus(BGE_COLLECTION, results_by_model.get("bge", []), SEARCH_DEPTH)
    fused_results = reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res}, MODEL_WEIGHTS)
    return add_image_urls(process_and_cluster_results(fused_results)[:TOP_K_RESULTS], str(request.base_url))

@app.post("/temporal_search")
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    models, stages_data, cluster_mode = request_data.models, request_data.stages, request_data.cluster
    if not stages_data: raise HTTPException(status_code=400, detail="No stages provided.")
    if not models: raise HTTPException(status_code=400, detail="No models selected.")
    
    print(f"Temporal Search | Cluster: {cluster_mode}, Stages: {len(stages_data)}, Models: {models}")

    async def get_stage_results(client, stage: StageDataBase):
        base_query = translate_text(stage.query)
        queries_to_process = expanding(base_query) if stage.expand else [base_query]
        queries_to_embed = [enhancing(q) for q in queries_to_process] if stage.enhance else queries_to_process
        tasks, model_order = [], []
        for model in models:
            url = BEIT3_WORKER_URL if model == "beit3" else (BGE_WORKER_URL if model == "bge" else None)
            if not url: continue
            for q in queries_to_embed:
                tasks.append(client.post(url, data={'text_query': q}))
                model_order.append(model)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        vecs_by_model = defaultdict(list)
        for i, resp in enumerate(responses):
            if not isinstance(resp, Exception) and resp.status_code == 200:
                vecs_by_model[model_order[i]].extend(resp.json()['embedding'])
        beit3_res = search_milvus(BEIT3_COLLECTION, vecs_by_model.get("beit3", []), SEARCH_DEPTH_PER_STAGE)
        bge_res = search_milvus(BGE_COLLECTION, vecs_by_model.get("bge", []), SEARCH_DEPTH_PER_STAGE)
        return reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res}, MODEL_WEIGHTS)

    async with httpx.AsyncClient(timeout=120.0) as client:
        all_stage_candidates = await asyncio.gather(*[get_stage_results(client, s) for s in stages_data])

    if cluster_mode:
        if len(all_stage_candidates) < 2: return []
        stage1_clusters = process_and_cluster_results(all_stage_candidates[0])
        stage2_clusters = process_and_cluster_results(all_stage_candidates[1])
        
        valid_sequences = []
        for c1 in stage1_clusters:
            for c2 in stage2_clusters:
                if c1['best_shot']['video_id'] == c2['best_shot']['video_id'] and \
                   c1['best_shot']['filepath'] < c2['best_shot']['filepath']:
                    avg_score = (c1['cluster_score'] + c2['cluster_score']) / 2
                    valid_sequences.append({
                        "average_score": avg_score,
                        "clusters": [c1, c2]
                    })
        
        if not valid_sequences: return []
        
        top_sequence = sorted(valid_sequences, key=lambda x: x['average_score'], reverse=True)[0]
        return add_image_urls(top_sequence['clusters'], str(request.base_url))
    else:
        results_by_video = defaultdict(lambda: defaultdict(list))
        for i, stage_candidates in enumerate(all_stage_candidates):
            for cand in stage_candidates:
                if cand.get('video_id'): results_by_video[cand['video_id']][i].append(cand)
        all_valid_sequences = []
        for video_id, video_stages in results_by_video.items():
            if len(video_stages) < len(stages_data): continue
            def find_combinations(current_sequence, stage_idx):
                if stage_idx == len(stages_data): all_valid_sequences.append(list(current_sequence)); return
                for next_cand in video_stages.get(stage_idx, []):
                    if not current_sequence or next_cand['filepath'] > current_sequence[-1]['filepath']:
                        current_sequence.append(next_cand); find_combinations(current_sequence, stage_idx + 1); current_sequence.pop()
            find_combinations([], 0)
        
        if not all_valid_sequences: return []
        processed_sequences = sorted([{"average_rrf_score": sum(s.get('rrf_score',0) for s in seq)/len(seq), "shots": seq} for seq in all_valid_sequences], key=lambda x: x['average_rrf_score'], reverse=True)
        return add_image_urls(processed_sequences[:MAX_SEQUENCES_TO_RETURN], str(request.base_url))

@app.post("/ocr_search")
async def ocr_search(request_data: OcrSearchRequest, request: Request):
    if not request_data.query: raise HTTPException(status_code=400, detail="Query is required for OCR search.")
    print(f"OCR Search | Enhance: {request_data.enhance}, Expand: {request_data.expand}")
    base_query = translate_text(request_data.query)
    queries_to_process = expanding(base_query) if request_data.expand else [base_query]
    queries_to_search = [enhancing(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    all_results, seen = [], set()
    for keyword in queries_to_search:
        for res in search_ocr_on_elasticsearch(keyword, limit=TOP_K_RESULTS):
            if res['filepath'] not in seen: all_results.append(res); seen.add(res['filepath'])
    sorted_results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:TOP_K_RESULTS]
    final_results = [{"best_shot": shot, "shots": [shot], "cluster_score": shot.get('score', 0)} for shot in sorted_results]
    return add_image_urls(final_results, str(request.base_url))

@app.post("/asr_search")
async def asr_search(request_data: AsrSearchRequest, request: Request):
    if not request_data.query: raise HTTPException(status_code=400, detail="Query is required for ASR search.")
    print(f"ASR Search | Enhance: {request_data.enhance}, Expand: {request_data.expand}")
    base_query = translate_text(request_data.query)
    queries_to_process = expanding(base_query) if request_data.expand else [base_query]
    queries_to_search = [enhancing(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    all_results, seen = [], set()
    for keyword in queries_to_search:
        for res in search_asr_on_elasticsearch(keyword, limit=TOP_K_RESULTS):
            if res['filepath'] not in seen: 
                all_results.append(res)
                seen.add(res['filepath'])
    sorted_results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:TOP_K_RESULTS]
    final_results = [{
        "best_shot": shot, "shots": [shot], "cluster_score": shot.get('score', 0)
    } for shot in sorted_results]
    return add_image_urls(final_results, str(request.base_url))

@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Invalid base64 path.")
    remapped_path = original_path.replace("/workspace", "/app", 1) if original_path.startswith("/workspace") else original_path
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(remapped_path)
    if not safe_path.startswith(safe_base) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    return FileResponse(safe_path)