# FILE: gateway_server.py (FINAL - With User's ADVANCED Temporal Filter Logic)

import os
import sys
import time
import traceback
import base64
import asyncio
import operator 
import json
from collections import defaultdict
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ValidationError
from pymilvus import Collection, connections, utility
from elasticsearch import Elasticsearch
import polars as pl 

# --- Setup and Pydantic Models (Unchanged) ---
_CURRENT_DIR_PARENT = os.path.dirname(os.path.abspath(__file__))
COMMON_PARENT_DIR = os.path.dirname(_CURRENT_DIR_PARENT)
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)
ALLOWED_BASE_DIR = "/app"
try:
    from translate_query import translate_text, enhancing, expanding
    print("--- Gateway Server: Đã import thành công các hàm xử lý truy vấn. ---")
except ImportError:
    print("!!! CẢNH BÁO: Không thể import các hàm xử lý truy vấn. Sử dụng hàm DUMMY. !!!")
    def enhancing(q: str) -> str:
        return q
    def expanding(q: str) -> list[str]:
        return [q]
    def translate_text(q: str) -> str:
        return q
BEIT3_WORKER_URL = "http://model-workers:8001/embed"
BGE_WORKER_URL = "http://model-workers:8002/embed"
ELASTICSEARCH_HOST = "http://elasticsearch2:9200"
OCR_ASR_INDEX_NAME = "opencubee_2"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
BEIT3_COLLECTION = "beit3_image_caption_embeddings"
BGE_COLLECTION = "bge_vl_large_image_embeddings"
MODEL_WEIGHTS = {"beit3": 0.6, "bge": 0.4}
SEARCH_DEPTH = 1000
TOP_K_RESULTS = 50
MAX_SEQUENCES_TO_RETURN = 20
SEARCH_DEPTH_PER_STAGE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720
es = None
app = FastAPI()
OBJECT_COUNTS_DF: Optional[pl.DataFrame] = None
OBJECT_POSITIONS_DF: Optional[pl.DataFrame] = None

class ObjectCountFilter(BaseModel):
    conditions: Dict[str, str] = {}

class PositionBox(BaseModel):
    label: str
    box: List[float]

class ObjectPositionFilter(BaseModel):
    boxes: List[PositionBox] = []

class ObjectFilters(BaseModel):
    counting: Optional[ObjectCountFilter] = None
    positioning: Optional[ObjectPositionFilter] = None

class StageDataBase(BaseModel):
    query: str
    expand: bool
    enhance: bool

class TemporalSearchRequest(BaseModel):
    stages: list[StageDataBase]
    models: List[str] = ["beit3", "bge"]
    cluster: bool = False
    filters: Optional[ObjectFilters] = None

class OcrSearchRequest(BaseModel):
    query: str
    expand: bool
    enhance: bool
    filters: Optional[ObjectFilters] = None

class AsrSearchRequest(BaseModel):
    query: str
    expand: bool
    enhance: bool
    filters: Optional[ObjectFilters] = None

class ProcessQueryRequest(BaseModel):
    query: str
    enhance: bool
    expand: bool

class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    models: List[str]
    enhance: bool = False
    expand: bool = False
    filters: Optional[ObjectFilters] = None

@app.on_event("startup")
def startup_event():
    global es, OBJECT_COUNTS_DF, OBJECT_POSITIONS_DF
    try: connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT); print("--- Milvus connection successful. ---")
    except Exception as e: print(f"FATAL: Could not connect to Milvus. Error: {e}")
    try: es = Elasticsearch(ELASTICSEARCH_HOST); es.ping(); print("--- Elasticsearch connection successful. ---")
    except Exception as e: print(f"FATAL: Could not connect to Elasticsearch. Error: {e}"); es = None
    try:
        print("--- Loading object detection data... ---")
        counts_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/HoangNguyen/support_script/inference_results_rfdetr_json/object_counts.parquet"
        positions_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/HoangNguyen/support_script/inference_results_rfdetr_json/object_positions.parquet"
        
        # --- MODIFIED: Create a 'name_stem' column by removing the extension ---
        counts_df = pl.read_parquet(counts_path)
        OBJECT_COUNTS_DF = counts_df.with_columns(
            pl.col("image_name").str.split(".").list.first().alias("name_stem")
        )

        positions_df = pl.read_parquet(positions_path)
        OBJECT_POSITIONS_DF = positions_df.with_columns([
            ((pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))).alias("bbox_area"),
            pl.col("image_name").str.split(".").list.first().alias("name_stem")
        ])
        print(f"--- Object data loaded. Counts: {OBJECT_COUNTS_DF.shape}, Positions: {OBJECT_POSITIONS_DF.shape} ---")
        print(f"DEBUG: Example name stem from counts: {OBJECT_COUNTS_DF.head(1)['name_stem'][0]}")

    except Exception as e:
        print(f"!!! WARNING: Could not load object parquet files. Filtering disabled. Error: {e} !!!")
        OBJECT_COUNTS_DF = None
        OBJECT_POSITIONS_DF = None

# --- NEW: Helper for Temporal Checklist Logic ---
def is_temporal_sequence_valid(sequence: Dict, filters: ObjectFilters) -> bool:
    print("\n--- [DEBUG] Checking Temporal Sequence ---")
    checklist = set()
    if filters.counting and filters.counting.conditions:
        for obj, cond in filters.counting.conditions.items(): checklist.add(f"count_{obj}_{cond}")
    if filters.positioning and filters.positioning.boxes:
        for i, pbox in enumerate(filters.positioning.boxes): checklist.add(f"pos_{i}_{pbox.label}")
    if not checklist: return True
    
    sequence_filepaths = {s['filepath'] for s in sequence.get('shots', []) if 'filepath' in s}
    for cluster in sequence.get('clusters', []):
        for shot in cluster.get('shots', []):
            if 'filepath' in shot: sequence_filepaths.add(shot['filepath'])
    if not sequence_filepaths: return False
    
    sequence_stems = {os.path.splitext(os.path.basename(p))[0] for p in sequence_filepaths}
    
    counts_subset = OBJECT_COUNTS_DF.filter(pl.col("name_stem").is_in(list(sequence_stems))) if filters.counting and OBJECT_COUNTS_DF is not None else None
    positions_subset = OBJECT_POSITIONS_DF.filter(pl.col("name_stem").is_in(list(sequence_stems))) if filters.positioning and OBJECT_POSITIONS_DF is not None else None

    for stem in sequence_stems:
        if not checklist: break 
        if counts_subset is not None:
            frame_counts = counts_subset.filter(pl.col("name_stem") == stem)
            if not frame_counts.is_empty():
                for obj, cond_str in filters.counting.conditions.items():
                    key = f"count_{obj}_{cond_str}"
                    if key in checklist:
                        op, val = parse_condition(cond_str)
                        if op and val is not None and obj in frame_counts.columns and op(frame_counts[0, obj], val):
                            checklist.remove(key)
        
        if positions_subset is not None:
            frame_positions = positions_subset.filter(pl.col("name_stem") == stem)
            if not frame_positions.is_empty():
                for i, p_box in enumerate(filters.positioning.boxes):
                    key = f"pos_{i}_{p_box.label}"
                    if key in checklist and p_box.label in frame_positions['object']:
                        # --- FINAL FIX: Use pl.when().then().otherwise() for universal Polars compatibility ---
                        user_x_min_lit = pl.lit(p_box.box[0] * IMAGE_WIDTH); user_y_min_lit = pl.lit(p_box.box[1] * IMAGE_HEIGHT)
                        user_x_max_lit = pl.lit(p_box.box[2] * IMAGE_WIDTH); user_y_max_lit = pl.lit(p_box.box[3] * IMAGE_HEIGHT)

                        intersect_x_min = pl.when(pl.col("x_min") > user_x_min_lit).then(pl.col("x_min")).otherwise(user_x_min_lit)
                        intersect_y_min = pl.when(pl.col("y_min") > user_y_min_lit).then(pl.col("y_min")).otherwise(user_y_min_lit)
                        intersect_x_max = pl.when(pl.col("x_max") < user_x_max_lit).then(pl.col("x_max")).otherwise(user_x_max_lit)
                        intersect_y_max = pl.when(pl.col("y_max") < user_y_max_lit).then(pl.col("y_max")).otherwise(user_y_max_lit)

                        intersect_area = (intersect_x_max - intersect_x_min).clip(lower_bound=0) * (intersect_y_max - intersect_y_min).clip(lower_bound=0)

                        match_df = frame_positions.filter(pl.col("object") == p_box.label).with_columns(
                            overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)
                        ).filter(pl.col("overlap_ratio") >= 0.75)
                        if not match_df.is_empty(): checklist.remove(key)
    
    is_valid = not checklist
    print(f"Result: {'VALID' if is_valid else 'INVALID'}. Remaining checklist: {checklist if not is_valid else 'None'}")
    return is_valid

# --- Object Filtering Logic (Unchanged but modularized) ---
def parse_condition(condition_str: str) -> tuple[Any, int]:
    try:
        return operator.ge, int(condition_str)
    except ValueError:
        op_map = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq, "=": operator.eq}
        for op_str in [">=", "<=", "==", ">", "<", "="]:
            if condition_str.startswith(op_str):
                try:
                    return op_map[op_str], int(condition_str[len(op_str):])
                except (ValueError, TypeError):
                    return None, None
    return None, None

def get_valid_filepaths_for_strict_search(all_filepaths: set, filters: ObjectFilters) -> set:
    print("\n--- [DEBUG] Applying STRICT Object Filters ---")
    print(f"Filter Conditions: {filters.dict()}")
    
    candidate_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_filepaths}
    print(f"Initial candidates: {len(all_filepaths)} (Unique name stems: {len(candidate_stems)})")
    if not candidate_stems: return set()
    print(f"Example candidate name stems: {list(candidate_stems)[:3]}")

    valid_stems = candidate_stems
    
    if filters.counting and OBJECT_COUNTS_DF is not None and filters.counting.conditions:
        df_subset = OBJECT_COUNTS_DF.filter(pl.col("name_stem").is_in(list(valid_stems)))
        expressions = []
        for obj, cond_str in filters.counting.conditions.items():
            op, val = parse_condition(cond_str)
            if op and val is not None and obj in df_subset.columns: expressions.append(op(pl.col(obj), val))
        
        if expressions: 
            valid_stems = set(df_subset.filter(pl.all_horizontal(expressions))['name_stem'])
            print(f"After COUNTING filter: {len(valid_stems)} candidates remain.")

    if filters.positioning and OBJECT_POSITIONS_DF is not None and filters.positioning.boxes:
        positions_subset_df = OBJECT_POSITIONS_DF.filter(pl.col("name_stem").is_in(list(valid_stems)))
        stems_satisfying_all_boxes = valid_stems
        for p_box in filters.positioning.boxes:
            # --- FINAL FIX: Use pl.when().then().otherwise() for universal Polars compatibility ---
            user_x_min_lit = pl.lit(p_box.box[0] * IMAGE_WIDTH); user_y_min_lit = pl.lit(p_box.box[1] * IMAGE_HEIGHT)
            user_x_max_lit = pl.lit(p_box.box[2] * IMAGE_WIDTH); user_y_max_lit = pl.lit(p_box.box[3] * IMAGE_HEIGHT)
            
            intersect_x_min = pl.when(pl.col("x_min") > user_x_min_lit).then(pl.col("x_min")).otherwise(user_x_min_lit)
            intersect_y_min = pl.when(pl.col("y_min") > user_y_min_lit).then(pl.col("y_min")).otherwise(user_y_min_lit)
            intersect_x_max = pl.when(pl.col("x_max") < user_x_max_lit).then(pl.col("x_max")).otherwise(user_x_max_lit)
            intersect_y_max = pl.when(pl.col("y_max") < user_y_max_lit).then(pl.col("y_max")).otherwise(user_y_max_lit)

            intersect_area = (intersect_x_max - intersect_x_min).clip(lower_bound=0) * (intersect_y_max - intersect_y_min).clip(lower_bound=0)

            condition_df = positions_subset_df.filter(pl.col("object") == p_box.label).with_columns(
                overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)
            ).filter(pl.col("overlap_ratio") >= 0.75)
            
            stems_satisfying_all_boxes = stems_satisfying_all_boxes.intersection(set(condition_df['name_stem'].unique()))
        
        valid_stems = stems_satisfying_all_boxes
        print(f"After POSITIONING filter: {len(valid_stems)} candidates remain.")

    print("--- [DEBUG] Strict Filtering Complete ---")
    return {fp for fp in all_filepaths if os.path.splitext(os.path.basename(fp))[0] in valid_stems}

# --- Support Functions (Unchanged) ---
def search_milvus(collection_name: str, query_vectors: list, limit: int, expr: str = None):
    try:
        if not utility.has_collection(collection_name) or not len(query_vectors):
            return []
        collection = Collection(collection_name)
        collection.load()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 128}}
        output_fields = ["filepath", "video_id", "shot_id", "frame_id"]
        results = collection.search(query_vectors, "image_embedding", search_params, limit=limit, output_fields=output_fields, expr=expr)
        parsed_results = []
        for s in results:
            for h in s:
                parsed_results.append({
                    "filepath": h.entity.get("filepath"),
                    "score": h.distance,
                    "video_id": h.entity.get("video_id"),
                    "frame_id": h.entity.get("frame_id"),
                    "shot_id": str(h.entity.get("shot_id"))
                })
        return parsed_results
    except Exception as e:
        print(f"ERROR during Milvus search on '{collection_name}': {e}")
        traceback.print_exc()
        return []

def convert_distance_to_similarity(results):
    for result in results:
        result['score'] = max(0, 1.0 - result.get('score', 1.0))
    return results

def reciprocal_rank_fusion(results_lists: dict, weights: dict, k_rrf: int = 60):
    master_data = defaultdict(lambda: {"raw_scores": {}})
    for model_name, results in results_lists.items():
        if not results:
            continue
        similarity_results = convert_distance_to_similarity(results)
        for rank, result in enumerate(similarity_results, 1):
            filepath = result.get('filepath')
            if not filepath:
                continue
            if 'metadata' not in master_data[filepath]:
                master_data[filepath]['metadata'] = result
            master_data[filepath]['raw_scores'][model_name] = {
                "score": result.get('score', 0.0),
                "rank": rank
            }
    if not master_data:
        return []
    final_results = []
    for filepath, data in master_data.items():
        rrf_score = 0.0
        for model_name, score_info in data['raw_scores'].items():
            rrf_score += weights.get(model_name, 1.0) * (1.0 / (k_rrf + score_info['rank']))
        final_item = data['metadata']
        final_item['rrf_score'] = rrf_score
        final_item.pop('score', None)
        final_results.append(final_item)
    return sorted(final_results, key=lambda x: x['rrf_score'], reverse=True)

def search_ocr_on_elasticsearch(keyword: str, limit: int=100):
    if not es:
        return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text"]}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
    except Exception as e:
        print(f"Lỗi Elasticsearch: {e}")
        return []
    results = []
    base_image_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/Enn/dataset/data_frame_ocr_png"
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        if not all([source.get('video'), source.get('shot_id'), source.get('frame')]):
            continue
        filename = f"{source['video']}_{source['shot_id']}_{str(source['frame']).zfill(6)}.png"
        results.append({
            "filepath": os.path.join(base_image_path, filename),
            "score": hit['_score'],
            "video_id": source.get('video'),
            "shot_id": source.get('shot_id')
        })
    return results

def search_asr_on_elasticsearch(keyword: str, limit: int=100):
    if not es:
        return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["asr_text"]}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
    except Exception as e:
        print(f"Lỗi Elasticsearch ASR: {e}")
        return []
    results = []
    base_image_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/Enn/dataset/data_frame_ocr_png"
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        if not all([source.get('video'), source.get('shot_id'), source.get('frame')]):
            continue
        filename = f"{source['video']}_{source['shot_id']}_{str(source['frame']).zfill(6)}.png"
        results.append({
            "filepath": os.path.join(base_image_path, filename),
            "score": hit['_score'],
            "video_id": source.get('video'),
            "shot_id": source.get('shot_id')
        })
    return results

def process_and_cluster_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    shots_by_video = defaultdict(list)
    for res in results:
        if not all(k in res for k in ['video_id', 'shot_id']):
            continue
        try:
            res['shot_id_int'] = int(str(res['shot_id']))
            shots_by_video[res['video_id']].append(res)
        except (ValueError, TypeError):
            continue
    all_clusters = []
    for video_id, shots in shots_by_video.items():
        if not shots:
            continue
        shots_by_shot_id = defaultdict(list)
        [shots_by_shot_id[shot['shot_id_int']].append(shot) for shot in shots]
        sorted_shot_ids = sorted(shots_by_shot_id.keys())
        if not sorted_shot_ids:
            continue
        current_cluster = []
        for i, shot_id in enumerate(sorted_shot_ids):
            if i > 0 and shot_id != sorted_shot_ids[i-1] + 1:
                if current_cluster:
                    all_clusters.append(current_cluster)
                current_cluster = []
            current_cluster.extend(shots_by_shot_id[shot_id])
        if current_cluster:
            all_clusters.append(current_cluster)
    if not all_clusters:
        return []
    processed_clusters = []
    for cluster_shots in all_clusters:
        if not cluster_shots:
            continue
        sorted_cluster_shots = sorted(cluster_shots, key=lambda x: x.get('rrf_score', x.get('score', 0)), reverse=True)
        best_shot = sorted_cluster_shots[0]
        max_score = best_shot.get('rrf_score', best_shot.get('score', 0))
        processed_clusters.append({
            "cluster_score": max_score,
            "shots": sorted_cluster_shots,
            "best_shot": best_shot
        })
    return sorted(processed_clusters, key=lambda x: x['cluster_score'], reverse=True)

def add_image_urls(data: List[Dict[str, Any]], base_url: str):
    for item in data:
        shots_list = item.get('shots', [])
        if not shots_list and 'clusters' in item:
             for cluster in item.get('clusters', []):
                 shots_list.extend(cluster.get('shots', []))
        for shot in shots_list:
             if shot.get('filepath') and 'url' not in shot:
                 shot['url'] = f"{base_url}images/{base64.urlsafe_b64encode(shot['filepath'].encode('utf-8')).decode('utf-8')}"
        if (best_shot := item.get('best_shot')) and best_shot.get('filepath') and 'url' not in best_shot:
            best_shot['url'] = f"{base_url}images/{base64.urlsafe_b64encode(best_shot['filepath'].encode('utf-8')).decode('utf-8')}"
    return data

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = "./ui1_2_temporal.html"
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=500, detail="UI file not found")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/process_query")
async def process_query(request_data: ProcessQueryRequest):
    if not request_data.query:
        return {"processed_query": ""}
    base_query = translate_text(request_data.query)
    queries_to_process = expanding(base_query) if request_data.expand else [base_query]
    final_queries = [enhancing(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    return {"processed_query": " ".join(final_queries)}

@app.post("/search")
async def search_unified(request: Request, search_data: str = Form(...), query_image: Optional[UploadFile] = File(None)):
    try:
        search_data_model = UnifiedSearchRequest.parse_raw(search_data)
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid search data format: {e}")
    if not search_data_model.query_text and not query_image:
        raise HTTPException(status_code=400, detail="Text or image query required.")
    if not search_data_model.models:
        raise HTTPException(status_code=400, detail="At least one model must be selected.")
    print(f"Simple Search | Filters: {search_data_model.filters}")
    final_queries_to_embed = []
    image_content = await query_image.read() if query_image else None
    if search_data_model.query_text:
        base_query = translate_text(search_data_model.query_text)
        queries_to_process = expanding(base_query) if search_data_model.expand else [base_query]
        final_queries_to_embed = [enhancing(q) for q in queries_to_process] if search_data_model.enhance else queries_to_process
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks, model_order = [], []
        for model in search_data_model.models:
            url = BEIT3_WORKER_URL if model == "beit3" else (BGE_WORKER_URL if model == "bge" else None)
            if not url:
                continue
            files = {'image_file': (query_image.filename, image_content, query_image.content_type)} if image_content else None
            if final_queries_to_embed:
                for q in final_queries_to_embed:
                    tasks.append(client.post(url, files=files, data={'text_query': q}))
                    model_order.append(model)
            elif files:
                tasks.append(client.post(url, files=files))
                model_order.append(model)
        if not tasks:
            raise HTTPException(status_code=400, detail="No valid query to process.")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results_by_model = defaultdict(list)
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"Error calling model {model_order[i]}: {resp}")
                continue
            if resp.status_code == 200:
                results_by_model[model_order[i]].extend(resp.json()['embedding'])
            else:
                print(f"Error from model {model_order[i]} (status {resp.status_code}): {resp.text}")
    beit3_res = search_milvus(BEIT3_COLLECTION, results_by_model.get("beit3", []), SEARCH_DEPTH)
    bge_res = search_milvus(BGE_COLLECTION, results_by_model.get("bge", []), SEARCH_DEPTH)
    fused_results = reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res}, MODEL_WEIGHTS)
    clustered_results = process_and_cluster_results(fused_results)
    
    # --- MODIFIED: Stricter filtering for simple search results ---
    if search_data_model.filters:
        all_filepaths = {s['filepath'] for c in clustered_results for s in c.get('shots', []) if 'filepath' in s}
        valid_filepaths = get_valid_filepaths_for_strict_search(all_filepaths, search_data_model.filters)
        # Rebuild clusters, only keeping valid shots
        final_results = []
        for cluster in clustered_results:
            filtered_shots = [s for s in cluster.get('shots', []) if s.get('filepath') in valid_filepaths]
            if filtered_shots:
                new_cluster = cluster.copy()
                new_cluster['shots'] = filtered_shots
                if new_cluster['best_shot'].get('filepath') not in valid_filepaths:
                    new_cluster['best_shot'] = max(filtered_shots, key=lambda x: x.get('rrf_score', 0))
                final_results.append(new_cluster)
    else:
        final_results = clustered_results
    
    return add_image_urls(final_results[:TOP_K_RESULTS], str(request.base_url))

@app.post("/temporal_search")
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    models, stages_data, cluster_mode, filters = request_data.models, request_data.stages, request_data.cluster, request_data.filters
    if not stages_data:
        raise HTTPException(status_code=400, detail="No stages provided.")
    if not models:
        raise HTTPException(status_code=400, detail="No models selected.")
    print(f"Temporal Search | Filters: {filters}")
    
    async def get_stage_results(client, stage: StageDataBase):
        base_query = translate_text(stage.query)
        queries_to_process = expanding(base_query) if stage.expand else [base_query]
        queries_to_embed = [enhancing(q) for q in queries_to_process] if stage.enhance else queries_to_process
        tasks, model_order = [], []
        for model in models:
            url = BEIT3_WORKER_URL if model == "beit3" else (BGE_WORKER_URL if model == "bge" else None)
            if not url:
                continue
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
        if len(all_stage_candidates) < 2:
            return []
        stage1_clusters = process_and_cluster_results(all_stage_candidates[0])
        stage2_clusters = process_and_cluster_results(all_stage_candidates[1])
        valid_sequences = []
        for c1 in stage1_clusters:
            for c2 in stage2_clusters:
                if c1['best_shot']['video_id'] == c2['best_shot']['video_id'] and c1['best_shot']['filepath'] < c2['best_shot']['filepath']:
                    avg_score = (c1['cluster_score'] + c2['cluster_score']) / 2
                    valid_sequences.append({"average_score": avg_score, "clusters": [c1, c2]})
        sequences_to_filter = sorted(valid_sequences, key=lambda x: x['average_score'], reverse=True)
    else:
        results_by_video = defaultdict(lambda: defaultdict(list))
        for i, stage_candidates in enumerate(all_stage_candidates):
            for cand in stage_candidates:
                if cand.get('video_id'):
                    results_by_video[cand['video_id']][i].append(cand)
        all_valid_sequences = []
        for video_id, video_stages in results_by_video.items():
            if len(video_stages) < len(stages_data):
                continue
                
            def find_combinations(current_sequence, stage_idx):
                if stage_idx == len(stages_data):
                    all_valid_sequences.append(list(current_sequence))
                    return
                for next_cand in video_stages.get(stage_idx, []):
                    if not current_sequence or next_cand['filepath'] > current_sequence[-1]['filepath']:
                        current_sequence.append(next_cand)
                        find_combinations(current_sequence, stage_idx + 1)
                        current_sequence.pop()
                        
            find_combinations([], 0)
        sequences_to_filter = sorted([{
            "average_rrf_score": sum(s.get('rrf_score',0) for s in seq)/len(seq),
            "shots": seq
        } for seq in all_valid_sequences], key=lambda x: x['average_rrf_score'], reverse=True)
    
    # --- THIS IS THE KEY CHANGE: APPLYING YOUR NEW CHECKLIST LOGIC ---
    if filters:
        filtered_sequences = [seq for seq in sequences_to_filter if is_temporal_sequence_valid(seq, filters)]
    else:
        filtered_sequences = sequences_to_filter
    
    return add_image_urls(filtered_sequences[:MAX_SEQUENCES_TO_RETURN], str(request.base_url))

@app.post("/ocr_search")
async def ocr_search(request_data: OcrSearchRequest, request: Request):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="Query is required for OCR search.")
    print(f"OCR Search | Filters: {request_data.filters}")
    base_query = translate_text(request_data.query)
    queries_to_process = expanding(base_query) if request_data.expand else [base_query]
    queries_to_search = [enhancing(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    all_results, seen = [], set()
    for keyword in queries_to_search:
        for res in search_ocr_on_elasticsearch(keyword, limit=TOP_K_RESULTS*2):
            if res['filepath'] not in seen:
                all_results.append(res)
                seen.add(res['filepath'])
    if request_data.filters:
        valid_filepaths = get_valid_filepaths_for_strict_search(seen, request_data.filters)
        filtered_results = [r for r in all_results if r['filepath'] in valid_filepaths]
    else:
        filtered_results = all_results
    sorted_results = sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=True)[:TOP_K_RESULTS]
    final_results = [{"best_shot": shot, "shots": [shot], "cluster_score": shot.get('score', 0)} for shot in sorted_results]
    return add_image_urls(final_results, str(request.base_url))

@app.post("/asr_search")
async def asr_search(request_data: AsrSearchRequest, request: Request):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="Query is required for ASR search.")
    print(f"ASR Search | Filters: {request_data.filters}")
    base_query = translate_text(request_data.query)
    queries_to_process = expanding(base_query) if request_data.expand else [base_query]
    queries_to_search = [enhancing(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    all_results, seen = [], set()
    for keyword in queries_to_search:
        for res in search_asr_on_elasticsearch(keyword, limit=TOP_K_RESULTS*2):
            if res['filepath'] not in seen:
                all_results.append(res)
                seen.add(res['filepath'])
    if request_data.filters:
        valid_filepaths = get_valid_filepaths_for_strict_search(seen, request_data.filters)
        filtered_results = [r for r in all_results if r['filepath'] in valid_filepaths]
    else:
        filtered_results = all_results
    sorted_results = sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=True)[:TOP_K_RESULTS]
    final_results = [{"best_shot": shot, "shots": [shot], "cluster_score": shot.get('score', 0)} for shot in sorted_results]
    return add_image_urls(final_results, str(request.base_url))

@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try:
        original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 path.")
    remapped_path = original_path.replace("/workspace", "/app", 1) if original_path.startswith("/workspace") else original_path
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(remapped_path)
    if not safe_path.startswith(safe_base) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    return FileResponse(safe_path)