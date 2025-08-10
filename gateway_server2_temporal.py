import os
import sys
import time
import traceback
import base64
import asyncio
import operator
import json
import re
from collections import defaultdict
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ValidationError
from pymilvus import Collection, connections, utility
from elasticsearch import Elasticsearch
import polars as pl
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory="."), name="static")

# --- Thiết lập đường dẫn & Import ---
_CURRENT_DIR_PARENT = os.path.dirname(os.path.abspath(__file__))
COMMON_PARENT_DIR = os.path.dirname(_CURRENT_DIR_PARENT)
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)
ALLOWED_BASE_DIR = "/app"

try:
    from translate_query import translate_query, enhance_query, expand_query_parallel
    print("--- Gateway Server: Đã import thành công các hàm xử lý truy vấn. ---")
except ImportError:
    print("!!! CẢNH BÁO: Không thể import các hàm xử lý truy vấn. Sử dụng hàm DUMMY. !!!")
    def enhance_query(q: str) -> str: return q
    def expand_query_parallel(q: str) -> list[str]: return [q]
    def translate_query(q: str) -> str: return q

# --- Cấu hình ---
BEIT3_WORKER_URL = "http://model-workers:8001/embed"
BGE_WORKER_URL = "http://model-workers:8002/embed"
UNITE_WORKER_URL = "http://model-workers:8003/embed"

ELASTICSEARCH_HOST = "http://elasticsearch2:9200"
OCR_ASR_INDEX_NAME = "opencubee_2"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"

BEIT3_COLLECTION = "beit3_image_embeddings_filtered"
BGE_COLLECTION = "bge_vl_large_image_embeddings_filtered"
UNITE_COLLECTION = "unite_qwen2_vl_sequential_embeddings_filtered"

MODEL_WEIGHTS = {"beit3": 0.4, "bge": 0.2, "unite": 0.4}

SEARCH_DEPTH = 1000
TOP_K_RESULTS = 50
MAX_SEQUENCES_TO_RETURN = 20
SEARCH_DEPTH_PER_STAGE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720

SEARCH_PARAMS = {
    "HNSW": {"metric_type": "COSINE", "params": {"nprobe": 128}},
    "IVF_FLAT": {"metric_type": "COSINE", "params": {"nprobe": 16}},
    "SCANN": {"metric_type": "COSINE", "params": {"nprobe": 128}},
    "DEFAULT": {"metric_type": "COSINE", "params": {}}
}
COLLECTION_TO_INDEX_TYPE = {
    BEIT3_COLLECTION: "SCANN",
    BGE_COLLECTION: "IVF_FLAT",
    UNITE_COLLECTION: "IVF_FLAT"
}

es = None
OBJECT_COUNTS_DF: Optional[pl.DataFrame] = None
OBJECT_POSITIONS_DF: Optional[pl.DataFrame] = None

# --- Pydantic Models ---
class ObjectCountFilter(BaseModel): conditions: Dict[str, str] = {}
class PositionBox(BaseModel): label: str; box: List[float]
class ObjectPositionFilter(BaseModel): boxes: List[PositionBox] = []
class ObjectFilters(BaseModel): counting: Optional[ObjectCountFilter] = None; positioning: Optional[ObjectPositionFilter] = None
class StageData(BaseModel): query: str; expand: bool; enhance: bool
class TemporalSearchRequest(BaseModel):
    stages: list[StageData]
    models: List[str] = ["beit3", "bge", "unite"]
    cluster: bool = False
    filters: Optional[ObjectFilters] = None
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
class ProcessQueryRequest(BaseModel): query: str; enhance: bool; expand: bool

class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    models: List[str] = ["beit3", "bge", "unite"]
    enhance: bool = False
    expand: bool = False
    filters: Optional[ObjectFilters] = None

class CheckFramesRequest(BaseModel):
    base_filepath: str

@app.on_event("startup")
def startup_event():
    global es, OBJECT_COUNTS_DF, OBJECT_POSITIONS_DF
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Milvus connection successful. ---")
    except Exception as e:
        print(f"FATAL: Could not connect to Milvus. Error: {e}")
    try:
        es = Elasticsearch(ELASTICSEARCH_HOST)
        if es.ping():
            print("--- Elasticsearch connection successful. ---")
        else:
            print("FATAL: Could not connect to Elasticsearch.")
            es = None
    except Exception as e:
        print(f"FATAL: Could not connect to Elasticsearch. Error: {e}")
        es = None
    try:
        print("--- Loading object detection data... ---")
        counts_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/HoangNguyen/support_script/inference_results_rfdetr_json/object_counts.parquet"
        positions_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/HoangNguyen/support_script/inference_results_rfdetr_json/object_positions.parquet"
        counts_df = pl.read_parquet(counts_path)
        OBJECT_COUNTS_DF = counts_df.with_columns(pl.col("image_name").str.split(".").list.first().alias("name_stem"))
        positions_df = pl.read_parquet(positions_path)
        OBJECT_POSITIONS_DF = positions_df.with_columns([((pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))).alias("bbox_area"), pl.col("image_name").str.split(".").list.first().alias("name_stem")])
        print(f"--- Object data loaded. Counts: {OBJECT_COUNTS_DF.shape}, Positions: {OBJECT_POSITIONS_DF.shape} ---")
    except Exception as e:
        print(f"!!! WARNING: Could not load object parquet files. Filtering disabled. Error: {e} !!!")
        OBJECT_COUNTS_DF = None; OBJECT_POSITIONS_DF = None

# --- Helper Functions ---
def get_filename_stem(filepath: str) -> Optional[str]:
    if not filepath: return None
    try: return os.path.splitext(os.path.basename(filepath))[0]
    except Exception: return None

def remap_filepaths_in_results(data: List[Dict[str, Any]]):
    if not isinstance(data, list): return data
    for item in data:
        if not isinstance(item, dict): continue
        def process_shot(shot_dict):
            if isinstance(shot_dict, dict) and 'filepath' in shot_dict:
                current_path = shot_dict.get('filepath')
                if current_path:
                    # First, handle the workspace to app remapping
                    if current_path.startswith("/workspace"):
                        current_path = current_path.replace("/workspace", "/app", 1)
                    
                    # Next, remove "_resized" from the path to get the original quality image path
                    current_path = current_path.replace("_resized", "")

                    shot_dict['filepath'] = current_path

        if 'shots' in item and isinstance(item['shots'], list):
            for shot in item['shots']: process_shot(shot)
        if 'best_shot' in item: process_shot(item['best_shot'])
        if 'clusters' in item and isinstance(item['clusters'], list):
            for cluster in item['clusters']:
                 if isinstance(cluster, dict):
                    if 'shots' in cluster and isinstance(cluster['shots'], list):
                        for shot in cluster['shots']: process_shot(shot)
                    if 'best_shot' in cluster: process_shot(cluster['best_shot'])
    return data

def is_temporal_sequence_valid(sequence: Dict, filters: ObjectFilters) -> bool:
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
    sequence_stems = {get_filename_stem(p) for p in sequence_filepaths}
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
                        user_x_min_lit, user_y_min_lit, user_x_max_lit, user_y_max_lit = [pl.lit(v) for v in [p_box.box[0] * IMAGE_WIDTH, p_box.box[1] * IMAGE_HEIGHT, p_box.box[2] * IMAGE_WIDTH, p_box.box[3] * IMAGE_HEIGHT]]
                        intersect_x_min = pl.when(pl.col("x_min") > user_x_min_lit).then(pl.col("x_min")).otherwise(user_x_min_lit)
                        intersect_y_min = pl.when(pl.col("y_min") > user_y_min_lit).then(pl.col("y_min")).otherwise(user_y_min_lit)
                        intersect_x_max = pl.when(pl.col("x_max") < user_x_max_lit).then(pl.col("x_max")).otherwise(user_x_max_lit)
                        intersect_y_max = pl.when(pl.col("y_max") < user_y_max_lit).then(pl.col("y_max")).otherwise(user_y_max_lit)
                        intersect_area = (intersect_x_max - intersect_x_min).clip(lower_bound=0) * (intersect_y_max - intersect_y_min).clip(lower_bound=0)
                        match_df = frame_positions.filter(pl.col("object") == p_box.label).with_columns(overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)).filter(pl.col("overlap_ratio") >= 0.75)
                        if not match_df.is_empty(): checklist.remove(key)
    return not checklist

def parse_condition(condition_str: str) -> tuple[Any, int]:
    try: return operator.ge, int(condition_str)
    except ValueError:
        op_map = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq, "=": operator.eq}
        for op_str in [">=", "<=", "==", ">", "<", "="]:
            if condition_str.startswith(op_str):
                try: return op_map[op_str], int(condition_str[len(op_str):])
                except (ValueError, TypeError): return None, None
    return None, None

def get_valid_filepaths_for_strict_search(all_filepaths: set, filters: ObjectFilters) -> set:
    candidate_stems = {get_filename_stem(p) for p in all_filepaths}
    if not candidate_stems: return set()
    valid_stems = candidate_stems
    if filters.counting and OBJECT_COUNTS_DF is not None and filters.counting.conditions:
        df_subset = OBJECT_COUNTS_DF.filter(pl.col("name_stem").is_in(list(valid_stems)))
        expressions = []
        for obj, cond_str in filters.counting.conditions.items():
            op, val = parse_condition(cond_str)
            if op and val is not None and obj in df_subset.columns: expressions.append(op(pl.col(obj), val))
        if expressions: valid_stems = set(df_subset.filter(pl.all_horizontal(expressions))['name_stem'])
    if filters.positioning and OBJECT_POSITIONS_DF is not None and filters.positioning.boxes:
        positions_subset_df = OBJECT_POSITIONS_DF.filter(pl.col("name_stem").is_in(list(valid_stems)))
        stems_satisfying_all_boxes = valid_stems
        for p_box in filters.positioning.boxes:
            user_x_min_lit, user_y_min_lit, user_x_max_lit, user_y_max_lit = [pl.lit(v) for v in [p_box.box[0] * IMAGE_WIDTH, p_box.box[1] * IMAGE_HEIGHT, p_box.box[2] * IMAGE_WIDTH, p_box.box[3] * IMAGE_HEIGHT]]
            intersect_x_min = pl.when(pl.col("x_min") > user_x_min_lit).then(pl.col("x_min")).otherwise(user_x_min_lit)
            intersect_y_min = pl.when(pl.col("y_min") > user_y_min_lit).then(pl.col("y_min")).otherwise(user_y_min_lit)
            intersect_x_max = pl.when(pl.col("x_max") < user_x_max_lit).then(pl.col("x_max")).otherwise(user_x_max_lit)
            intersect_y_max = pl.when(pl.col("y_max") < user_y_max_lit).then(pl.col("y_max")).otherwise(user_y_max_lit)
            intersect_area = (intersect_x_max - intersect_x_min).clip(lower_bound=0) * (intersect_y_max - intersect_y_min).clip(lower_bound=0)
            condition_df = positions_subset_df.filter(pl.col("object") == p_box.label).with_columns(overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)).filter(pl.col("overlap_ratio") >= 0.75)
            stems_satisfying_all_boxes = stems_satisfying_all_boxes.intersection(set(condition_df['name_stem'].unique()))
        valid_stems = stems_satisfying_all_boxes
    return {fp for fp in all_filepaths if get_filename_stem(fp) in valid_stems}

def search_milvus(collection_name: str, query_vectors: list, limit: int, expr: str = None):
    try:
        if not utility.has_collection(collection_name) or not len(query_vectors): return []
        collection = Collection(collection_name)
        collection.load()
        index_type = COLLECTION_TO_INDEX_TYPE.get(collection_name, "DEFAULT")
        search_params = SEARCH_PARAMS.get(index_type, SEARCH_PARAMS["DEFAULT"])
        anns_field = "image_embedding"
        output_fields = ["filepath", "video_id", "shot_id", "frame_id"]
        results = collection.search(query_vectors, anns_field, search_params, limit=limit, output_fields=output_fields, expr=expr)
        parsed_results = []
        for s in results:
            for h in s:
                parsed_results.append({
                    "filepath": h.entity.get("filepath"), "score": h.distance,
                    "video_id": h.entity.get("video_id"), "frame_id": h.entity.get("frame_id"),
                    "shot_id": str(h.entity.get("shot_id"))
                })
        return parsed_results
    except Exception as e:
        print(f"ERROR during Milvus search on '{collection_name}': {e}"); traceback.print_exc()
        return []

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
        final_item = data['metadata']
        final_item['rrf_score'] = rrf_score
        final_item.pop('score', None)
        final_results.append(final_item)
    return sorted(final_results, key=lambda x: x['rrf_score'], reverse=True)

def search_ocr_on_elasticsearch(keyword: str, limit: int = 500):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text"]}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit['_source']
            if all(k in source for k in ['file_path', 'video_id', 'shot_id', 'frame_id']):
                results.append({"filepath": source['file_path'], "score": hit['_score'], "video_id": source['video_id'], "shot_id": str(source['shot_id']), "frame_id": source['frame_id']})
        return results
    except Exception as e:
        print(f"Lỗi Elasticsearch OCR: {e}"); return []

def search_asr_on_elasticsearch(keyword: str, limit: int = 500):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["asr_text^3", "text^1"], "type": "best_fields", "fuzziness": "AUTO"}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit['_source']
            filepath = source.get('file_path') or source.get('filepath')
            if filepath and all(k in source for k in ['video_id', 'shot_id', 'frame_id']):
                results.append({"filepath": filepath, "score": hit['_score'], "video_id": source['video_id'], "shot_id": str(source['shot_id']), "frame_id": source['frame_id']})
        return results
    except Exception as e:
        print(f"Lỗi Elasticsearch ASR: {e}"); return []

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
        shots_by_shot_id = defaultdict(list)
        for shot in shots: shots_by_shot_id[shot['shot_id_int']].append(shot)
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
    if not isinstance(data, list): return data
    for item in data:
        if not isinstance(item, dict): continue
        def process_shot(shot_dict):
            if isinstance(shot_dict, dict) and shot_dict.get('filepath') and 'url' not in shot_dict:
                shot_dict['url'] = f"{base_url}images/{base64.urlsafe_b64encode(shot_dict['filepath'].encode('utf-8')).decode('utf-8')}"
        if 'shots' in item and isinstance(item['shots'], list):
            for shot in item['shots']: process_shot(shot)
        if 'best_shot' in item: process_shot(item['best_shot'])
        if 'clusters' in item and isinstance(item['clusters'], list):
            for cluster in item['clusters']:
                 if isinstance(cluster, dict):
                    if 'shots' in cluster and isinstance(cluster['shots'], list):
                        for shot in cluster['shots']: process_shot(shot)
                    if 'best_shot' in cluster: process_shot(cluster['best_shot'])
    return data

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = os.path.join(BASE_DIR, "ui1_2_temporal.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=500, detail="UI file not found")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/process_query")
async def process_query(request_data: ProcessQueryRequest):
    if not request_data.query: return {"processed_query": ""}
    base_query = translate_query(request_data.query)
    queries_to_process = expand_query_parallel(base_query) if request_data.expand else [base_query]
    final_queries = [enhance_query(q) for q in queries_to_process] if request_data.enhance else queries_to_process
    return {"processed_query": " ".join(final_queries)}

@app.post("/search")
async def search_unified(request: Request, search_data: str = Form(...), query_image: Optional[UploadFile] = File(None)):
    try:
        search_data_model = UnifiedSearchRequest.parse_raw(search_data)
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid search data format: {e}")

    is_primary_search = bool(search_data_model.query_text or query_image)
    is_filter_search = bool(search_data_model.ocr_query or search_data_model.asr_query)

    if not is_primary_search and not is_filter_search:
        raise HTTPException(status_code=400, detail="Text, image, OCR, or ASR query required.")

    milvus_results, es_results = [], []
    
    if is_primary_search:
        if not search_data_model.models: raise HTTPException(status_code=400, detail="At least one model must be selected for primary search.")
        
        final_queries_to_embed = []
        if search_data_model.query_text:
            base_query = translate_query(search_data_model.query_text)
            queries_to_process = expand_query_parallel(base_query) if search_data_model.expand else [base_query]
            final_queries_to_embed = [enhance_query(q) for q in queries_to_process] if search_data_model.enhance else queries_to_process

        image_content = await query_image.read() if query_image else None
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks, model_order = [], []
            model_url_map = {"beit3": BEIT3_WORKER_URL, "bge": BGE_WORKER_URL, "unite": UNITE_WORKER_URL}
            for model in search_data_model.models:
                url = model_url_map.get(model)
                if not url: continue
                can_process_image = (model != "bge_m3") and image_content
                files = {'image_file': (query_image.filename, image_content, query_image.content_type)} if can_process_image else None
                if final_queries_to_embed:
                    for q in final_queries_to_embed:
                        tasks.append(client.post(url, files=files, data={'text_query': q}))
                        model_order.append(model)
                elif can_process_image:
                    tasks.append(client.post(url, files=files))
                    model_order.append(model)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
            results_by_model = defaultdict(list)
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception): print(f"Error calling model {model_order[i]}: {resp}"); continue
                if resp.status_code == 200: results_by_model[model_order[i]].extend(resp.json()['embedding'])
                else: print(f"Error from model {model_order[i]} (status {resp.status_code}): {resp.text}")
        
        if any(results_by_model.values()):
            beit3_res = search_milvus(BEIT3_COLLECTION, results_by_model.get("beit3", []), SEARCH_DEPTH)
            bge_res = search_milvus(BGE_COLLECTION, results_by_model.get("bge", []), SEARCH_DEPTH)
            unite_res = search_milvus(UNITE_COLLECTION, results_by_model.get("unite", []), SEARCH_DEPTH)
            
            milvus_weights = {m: w for m, w in MODEL_WEIGHTS.items() if m in search_data_model.models}
            milvus_results = reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res, "unite": unite_res}, milvus_weights)

    if is_filter_search:
        ocr_res = search_ocr_on_elasticsearch(search_data_model.ocr_query, limit=SEARCH_DEPTH * 2) if search_data_model.ocr_query else []
        asr_res = search_asr_on_elasticsearch(search_data_model.asr_query, limit=SEARCH_DEPTH * 2) if search_data_model.asr_query else []
        
        es_res_map = {res['filepath']: res for res in ocr_res}
        for res in asr_res:
            if res['filepath'] not in es_res_map: es_res_map[res['filepath']] = res
        es_results = list(es_res_map.values())

    final_fused_results = []
    if is_primary_search and is_filter_search:
        if milvus_results and es_results:
            es_stems = {get_filename_stem(r['filepath']) for r in es_results if r.get('filepath')}
            final_fused_results = [r for r in milvus_results if get_filename_stem(r.get('filepath')) in es_stems]
    elif is_primary_search:
        final_fused_results = milvus_results
    elif is_filter_search:
        for res in es_results: res['rrf_score'] = res.pop('score', 0.0)
        final_fused_results = sorted(es_results, key=lambda x: x.get('rrf_score', 0), reverse=True)
    
    clustered_results = process_and_cluster_results(final_fused_results)

    final_results = clustered_results
    if search_data_model.filters and clustered_results:
        all_filepaths = {s['filepath'] for c in clustered_results for s in c.get('shots', []) if 'filepath' in s}
        valid_filepaths = get_valid_filepaths_for_strict_search(all_filepaths, search_data_model.filters)
        final_results = []
        for cluster in clustered_results:
            filtered_shots = [s for s in cluster.get('shots', []) if s.get('filepath') in valid_filepaths]
            if filtered_shots:
                new_cluster = cluster.copy()
                new_cluster['shots'] = filtered_shots
                if new_cluster['best_shot'].get('filepath') not in valid_filepaths:
                    new_cluster['best_shot'] = max(filtered_shots, key=lambda x: x.get('rrf_score', 0))
                final_results.append(new_cluster)
    
    final_results = remap_filepaths_in_results(final_results)
    return add_image_urls(final_results[:TOP_K_RESULTS], str(request.base_url))

@app.post("/temporal_search")
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    models, stages, filters = request_data.models, request_data.stages, request_data.filters
    ocr_query, asr_query = request_data.ocr_query, request_data.asr_query
    is_filter_search = bool(ocr_query or asr_query)

    if not stages: raise HTTPException(status_code=400, detail="No stages provided.")
    if not models: raise HTTPException(status_code=400, detail="No models selected.")
    
    # --- Bước 1: Lấy kết quả ứng viên từ Milvus (Giữ nguyên) ---
    async def get_stage_results(client, stage: StageData):
        base_query = translate_query(stage.query)
        queries_to_process = expand_query_parallel(base_query) if stage.expand else [base_query]
        queries_to_embed = [enhance_query(q) for q in queries_to_process] if stage.enhance else queries_to_process
        tasks, model_order = [], []
        model_url_map = {"beit3": BEIT3_WORKER_URL, "bge": BGE_WORKER_URL, "unite": UNITE_WORKER_URL}
        for model in models:
            url = model_url_map.get(model)
            if not url: continue
            tasks.extend([client.post(url, data={'text_query': q}) for q in queries_to_embed])
            model_order.extend([model] * len(queries_to_embed))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        vecs_by_model = defaultdict(list)
        for i, resp in enumerate(responses):
            if not isinstance(resp, Exception) and resp.status_code == 200:
                vecs_by_model[model_order[i]].extend(resp.json()['embedding'])
        beit3_res = search_milvus(BEIT3_COLLECTION, vecs_by_model.get("beit3", []), SEARCH_DEPTH_PER_STAGE)
        bge_res = search_milvus(BGE_COLLECTION, vecs_by_model.get("bge", []), SEARCH_DEPTH_PER_STAGE)
        unite_res = search_milvus(UNITE_COLLECTION, vecs_by_model.get("unite", []), SEARCH_DEPTH_PER_STAGE)
        return reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res, "unite": unite_res}, MODEL_WEIGHTS)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        all_stage_candidates = await asyncio.gather(*[get_stage_results(client, stage) for stage in stages])

    # --- Bước 2: Tìm các chuỗi hợp lệ (Giữ nguyên) ---
    clustered_results_by_stage = [process_and_cluster_results(res) for res in all_stage_candidates]
    for stage_clusters in clustered_results_by_stage:
        for cluster in stage_clusters:
            if cluster.get('shots'):
                shot_ids_int = [s['shot_id_int'] for s in cluster['shots'] if 'shot_id_int' in s]
                if shot_ids_int: 
                    cluster['min_shot_id'] = min(shot_ids_int); cluster['max_shot_id'] = max(shot_ids_int)
                    cluster['video_id'] = cluster['best_shot']['video_id']
    clusters_by_video = defaultdict(lambda: defaultdict(list))
    for i, stage_clusters in enumerate(clustered_results_by_stage):
        for cluster in stage_clusters:
            if 'video_id' in cluster: clusters_by_video[cluster['video_id']][i].append(cluster)
    all_valid_cluster_sequences = []
    for video_id, video_stages in clusters_by_video.items():
        if len(video_stages) < len(stages): continue
        def find_cluster_combinations(current_sequence, stage_idx):
            if stage_idx == len(stages): all_valid_cluster_sequences.append(list(current_sequence)); return
            for next_cluster in video_stages.get(stage_idx, []):
                if not current_sequence or next_cluster.get('min_shot_id', -1) > current_sequence[-1].get('max_shot_id', -1):
                    current_sequence.append(next_cluster); find_cluster_combinations(current_sequence, stage_idx + 1); current_sequence.pop()
        find_cluster_combinations([], 0)
    if not all_valid_cluster_sequences: return []
    processed_sequences = []
    for cluster_seq in all_valid_cluster_sequences:
        if not cluster_seq: continue
        avg_score = sum(c.get('cluster_score', 0) for c in cluster_seq) / len(cluster_seq)
        processed_sequences.append({"average_rrf_score": avg_score, "clusters": cluster_seq, "shots": [c['best_shot'] for c in cluster_seq], "video_id": cluster_seq[0].get('video_id', 'N/A')})
    
    sequences_to_filter = sorted(processed_sequences, key=lambda x: x['average_rrf_score'], reverse=True)
    
    # --- Bước 3: Áp dụng các bộ lọc (LOGIC MỚI) ---
    final_sequences = []
    
    # Chuẩn bị bộ lọc ES nếu có
    es_stems = set()
    if is_filter_search:
        ocr_res = search_ocr_on_elasticsearch(ocr_query, limit=SEARCH_DEPTH * 5) if ocr_query else []
        asr_res = search_asr_on_elasticsearch(asr_query, limit=SEARCH_DEPTH * 5) if asr_query else []
        es_filepaths = {r['filepath'] for r in ocr_res} | {r['filepath'] for r in asr_res}
        es_stems = {get_filename_stem(fp) for fp in es_filepaths}

    # Lặp qua các chuỗi và kiểm tra tất cả điều kiện lọc
    for seq in sequences_to_filter:
        # Điều kiện 1: Lọc bằng Object Filter (giữ nguyên)
        if filters and not is_temporal_sequence_valid(seq, filters):
            continue
        
        # Điều kiện 2: Lọc bằng OCR/ASR (mới)
        if is_filter_search:
            sequence_stems = {get_filename_stem(s['filepath']) for s in seq.get('shots', [])}
            # Yêu cầu: ít nhất một frame trong chuỗi phải khớp với bộ lọc OCR/ASR
            if not sequence_stems.intersection(es_stems):
                continue
        
        # Nếu qua tất cả các bộ lọc, thêm vào kết quả cuối cùng
        final_sequences.append(seq)
        
    final_sequences_remapped = remap_filepaths_in_results(final_sequences)
    return add_image_urls(final_sequences_remapped[:MAX_SEQUENCES_TO_RETURN], str(request.base_url))

@app.post("/check_temporal_frames")
async def check_temporal_frames(request_data: CheckFramesRequest) -> List[str]:
    base_filepath = request_data.base_filepath
    if not base_filepath or not os.path.isfile(base_filepath):
        raise HTTPException(status_code=404, detail="Base filepath not found or does not exist.")
    try:
        directory = os.path.dirname(base_filepath)
        target_filename = os.path.basename(base_filepath)
        video_match = re.match(r'^(L\d+_V\d+)', target_filename)
        if not video_match: return [base_filepath]
        video_prefix = video_match.group(1)
        all_frames_in_video = []
        for filename in os.listdir(directory):
            if filename.startswith(video_prefix):
                frame_num_match = re.search(r'_(\d+)\.[^.]+$', filename)
                if frame_num_match:
                    all_frames_in_video.append({'num': int(frame_num_match.group(1)), 'path': os.path.join(directory, filename)})
        all_frames_in_video.sort(key=lambda x: x['num'])
        sorted_paths = [frame['path'] for frame in all_frames_in_video]
        try: target_index = sorted_paths.index(base_filepath)
        except ValueError: return [base_filepath]
        start_index = max(0, target_index - 10)
        end_index = min(len(sorted_paths), target_index + 11)
        return sorted_paths[start_index:end_index]
    except Exception as e:
        print(f"ERROR in check_temporal_frames: {e}"); traceback.print_exc()
        return []

@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Invalid base64 path.")
    remapped_path = original_path.replace("/workspace", "/app", 1) if original_path.startswith("/workspace") else original_path
    safe_base, safe_path = os.path.realpath(ALLOWED_BASE_DIR), os.path.realpath(remapped_path)
    if not safe_path.startswith(safe_base) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    return FileResponse(safe_path)