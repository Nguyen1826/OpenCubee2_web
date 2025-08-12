import os
import sys
import time
import traceback
import base64
import asyncio
import operator
import json
import re
import uuid
import shutil
from pathlib import Path
from collections import defaultdict
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError
from pymilvus import Collection, connections, utility
import requests # Thêm import này
try:
    # Newer versions
    from pymilvus import LoadState as _MilvusLoadState
except ImportError:
    try:
        # Some versions keep it in client.constants
        from pymilvus.client.constants import LoadState as _MilvusLoadState
    except Exception:
        _MilvusLoadState = None
from elasticsearch import Elasticsearch, NotFoundError
import polars as pl
from fastapi.staticfiles import StaticFiles

# --- Thiết lập & Cấu hình ---
app = FastAPI()
BASE_DIR = os.path.dirname(__file__)
# NOTE: Changed from "static" to "." to serve logo.png from the root directory where the script is.
app.mount("/static", StaticFiles(directory="."), name="static")


TEMP_UPLOAD_DIR = Path("/app/temp_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

_CURRENT_DIR_PARENT = os.path.dirname(os.path.abspath(__file__))
COMMON_PARENT_DIR = os.path.dirname(_CURRENT_DIR_PARENT)
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)
ALLOWED_BASE_DIR = "/app"

try:
    from function import translate_query, enhance_query, expand_query_parallel
    print("--- Gateway Server: Đã import thành công các hàm xử lý truy vấn. ---")
except ImportError:
    print("!!! CẢNH BÁO: Không thể import các hàm xử lý truy vấn. Sử dụng hàm DUMMY. !!!")
    def enhance_query(q: str) -> str: return q
    def expand_query_parallel(q: str) -> list[str]: return [q]
    # Dummy function must be async to match the real one
    async def translate_query(q: str) -> str: return q

# --- Cấu hình DRES và hệ thống ---
DRES_BASE_URL = "http://192.168.28.151:5000"
VIDEO_FPS = 25 

BEIT3_WORKER_URL = "http://model-workers:8001/embed"
BGE_WORKER_URL = "http://model-workers:8002/embed"
UNITE_WORKER_URL = "http://model-workers:8003/embed"
IMAGE_GEN_WORKER_URL = "http://localhost:8004/generate"

ELASTICSEARCH_HOST = "http://elasticsearch2:9200"
OCR_ASR_INDEX_NAME = "opencubee_2"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"

BEIT3_COLLECTION_NAME = "beit3_image_embeddings_filtered"
BGE_COLLECTION_NAME = "bge_vl_large_image_embeddings_filtered"
UNITE_COLLECTION_NAME = "unite_qwen2_vl_sequential_embeddings_filtered"

MODEL_WEIGHTS = {"beit3": 0.4, "bge": 0.2, "unite": 0.4}
SEARCH_DEPTH = 1000
TOP_K_RESULTS = 50
MAX_SEQUENCES_TO_RETURN = 50
SEARCH_DEPTH_PER_STAGE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720

SEARCH_PARAMS = {
    "HNSW": {"metric_type": "COSINE", "params": {"nprobe": 128}},
    "IVF_FLAT": {"metric_type": "COSINE", "params": {"nprobe": 16}},
    "SCANN": {"metric_type": "COSINE", "params": {"nprobe": 128}},
    "DEFAULT": {"metric_type": "COSINE", "params": {}}
}
COLLECTION_TO_INDEX_TYPE = { 
    BEIT3_COLLECTION_NAME: "SCANN", 
    BGE_COLLECTION_NAME: "IVF_FLAT", 
    UNITE_COLLECTION_NAME: "IVF_FLAT" 
}

es = None
OBJECT_COUNTS_DF: Optional[pl.DataFrame] = None
OBJECT_POSITIONS_DF: Optional[pl.DataFrame] = None
beit3_collection: Optional[Collection] = None
bge_collection: Optional[Collection] = None
unite_collection: Optional[Collection] = None

# --- Pydantic Models ---
class ObjectCountFilter(BaseModel): conditions: Dict[str, str] = {}
class PositionBox(BaseModel): label: str; box: List[float]
class ObjectPositionFilter(BaseModel): boxes: List[PositionBox] = []
class ObjectFilters(BaseModel): counting: Optional[ObjectCountFilter] = None; positioning: Optional[ObjectPositionFilter] = None

# --- CHANGE: Added query_image_name to StageData ---
class StageData(BaseModel): 
    query: str
    enhance: bool
    expand: bool
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    query_image_name: Optional[str] = None # <-- FIELD MỚI
    generated_image_name: Optional[str] = None

class TemporalSearchRequest(BaseModel):
    stages: list[StageData]
    models: List[str] = ["beit3", "bge", "unite"]
    cluster: bool = False
    filters: Optional[ObjectFilters] = None
class ProcessQueryRequest(BaseModel): 
    query: str
    enhance: bool = False
    expand: bool = False
class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_image_name: Optional[str] = None
    image_search_text: Optional[str] = None
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    models: List[str] = ["beit3", "bge", "unite"]
    filters: Optional[ObjectFilters] = None
    enhance: bool = False
    expand: bool = False
    generated_image_name: Optional[str] = None
class CheckFramesRequest(BaseModel): base_filepath: str
class DRESLoginRequest(BaseModel): username: str; password: str
class DRESSubmitRequest(BaseModel):
    sessionId: str
    evaluationId: str
    video_id: str
    filepath: str

@app.on_event("startup")
def startup_event():
    global es, OBJECT_COUNTS_DF, OBJECT_POSITIONS_DF, beit3_collection, bge_collection, unite_collection
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Milvus connection successful. ---")
        print("--- Loading Milvus collections into memory... ---")
        collections_to_load = {"BEiT3": BEIT3_COLLECTION_NAME, "BGE": BGE_COLLECTION_NAME, "Unite": UNITE_COLLECTION_NAME}
        for name, col_name in collections_to_load.items():
            if utility.has_collection(col_name):
                collection = Collection(col_name)
                collection.load()
                if name == "BEiT3": beit3_collection = collection
                elif name == "BGE": bge_collection = collection
                elif name == "Unite": unite_collection = collection
                print(f"--- Collection '{col_name}' loaded successfully. ---")
            else: print(f"!!! WARNING: Collection '{col_name}' not found. !!!")
    except Exception as e: print(f"FATAL: Could not connect to or load from Milvus. Error: {e}")
    try:
        es = Elasticsearch(ELASTICSEARCH_HOST)
        if es.ping(): print("--- Elasticsearch connection successful. ---")
        else: print("FATAL: Could not connect to Elasticsearch."); es = None
    except Exception as e: print(f"FATAL: Could not connect to Elasticsearch. Error: {e}"); es = None
    try:
        print("--- Loading object detection data... ---")
        counts_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/HoangNguyen/support_script/inference_results_rfdetr_json/object_counts.parquet"
        positions_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/HoangNguyen/support_script/inference_results_rfdetr_json/object_positions.parquet"
        counts_df = pl.read_parquet(counts_path)
        OBJECT_COUNTS_DF = counts_df.with_columns(pl.col("image_name").str.split(".").list.first().alias("name_stem"))
        positions_df = pl.read_parquet(positions_path)
        OBJECT_POSITIONS_DF = positions_df.with_columns([((pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))).alias("bbox_area"), pl.col("image_name").str.split(".").list.first().alias("name_stem")])
        print(f"--- Object data loaded. ---")
    except Exception as e:
        print(f"!!! WARNING: Could not load object parquet files. Filtering disabled. Error: {e} !!!"); OBJECT_COUNTS_DF = None; OBJECT_POSITIONS_DF = None

# --- Helper Functions ---
def process_and_cluster_results_optimized(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optimized version of the clustering function using a "Sort and Iterate" approach.
    This version is complete and returns data in the same format as the original function.
    """
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
    for video_id, shots_in_video in shots_by_video.items():
        if not shots_in_video:
            continue

        # 1. Sort all shots for this video by their shot_id
        sorted_shots = sorted(shots_in_video, key=lambda x: x['shot_id_int'])

        if not sorted_shots:
            continue

        # 2. Iterate once through the sorted list to form raw clusters (lists of shots)
        current_cluster = [sorted_shots[0]]
        for i in range(1, len(sorted_shots)):
            current_shot = sorted_shots[i]
            last_shot_in_cluster = current_cluster[-1]
            
            if current_shot['shot_id_int'] == last_shot_in_cluster['shot_id_int']:
                current_cluster.append(current_shot)
            elif current_shot['shot_id_int'] == last_shot_in_cluster['shot_id_int'] + 1:
                current_cluster.append(current_shot)
            else:
                all_clusters.append(current_cluster)
                current_cluster = [current_shot]

        if current_cluster:
            all_clusters.append(current_cluster)

    if not all_clusters:
        return []
        
    # --- THIS IS THE CRUCIAL MISSING PART ---
    # 3. Process the finalized raw clusters into the correct dictionary format
    processed_clusters = []
    for cluster_shots in all_clusters:
        if not cluster_shots:
            continue
        
        # Sort shots within the cluster by score to find the best one
        sorted_cluster_shots = sorted(
            cluster_shots, 
            key=lambda x: x.get('rrf_score', x.get('score', 0)), 
            reverse=True
        )
        best_shot = sorted_cluster_shots[0]
        max_score = best_shot.get('rrf_score', best_shot.get('score', 0))
        
        processed_clusters.append({
            "cluster_score": max_score,
            "shots": sorted_cluster_shots,
            "best_shot": best_shot
        })
    # --- END OF THE MISSING PART ---

    # 4. Sort all found processed clusters by their final score
    return sorted(processed_clusters, key=lambda x: x['cluster_score'], reverse=True)
        

def get_filename_stem(filepath: str) -> Optional[str]:
    if not filepath: return None
    try: return os.path.splitext(os.path.basename(filepath))[0]
    except Exception: return None

# --- TỐI ƯU ---
# Hàm này vẫn là hàm đồng bộ (synchronous) vì logic Polars bên trong không phải là async.
# Tuy nhiên, chúng ta sẽ gọi nó bằng `asyncio.to_thread` để tránh block event loop.
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

# --- TỐI ƯU ---
# Hàm này cũng sẽ được gọi bằng `asyncio.to_thread` vì nó có thể tốn thời gian
# khi xử lý một lượng lớn filepaths và các bộ lọc phức tạp.
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

# --- I/O & CPU Functions ---
def search_milvus_sync(collection: Collection, collection_name: str, query_vectors: list, limit: int, expr: str = None):
    try:
        if not collection or not query_vectors:
            return []

        need_load = False
        try:
            state = utility.load_state(collection.name)
            if _MilvusLoadState and isinstance(state, _MilvusLoadState):
                need_load = (state != _MilvusLoadState.Loaded)
            else:
                state_name = getattr(state, "name", str(state))
                if str(state_name).lower() != "loaded" and str(state) != "2":
                    need_load = True
        except Exception as e:
            print(f"Could not determine load state for '{collection.name}' ({e}); will attempt load.")
            need_load = True

        if need_load:
            print(f"--- Collection '{collection.name}' not loaded. Loading... ---")
            collection.load()
            print(f"--- Collection '{collection.name}' loaded. ---")

        index_type = COLLECTION_TO_INDEX_TYPE.get(collection_name, "DEFAULT")
        search_params = SEARCH_PARAMS.get(index_type, SEARCH_PARAMS["DEFAULT"])

        output_fields = ["filepath", "video_id", "shot_id", "frame_id"]
        try:
            if any(f.name == "name_stem" for f in collection.schema.fields):
                output_fields.append("name_stem")
        except Exception:
            pass

        results = collection.search(
            data=query_vectors,
            anns_field="image_embedding",
            param=search_params,
            limit=limit,
            output_fields=output_fields,
            expr=expr
        )

        return [
            {
                "filepath": hit.entity.get("filepath"),
                "score": hit.distance,
                "video_id": hit.entity.get("video_id"),
                "frame_id": hit.entity.get("frame_id"),
                "shot_id": str(hit.entity.get("shot_id"))
            }
            for one_query in results for hit in one_query
        ]
    except Exception as e:
        print(f"ERROR during Milvus search on '{collection_name}': {e}")
        traceback.print_exc()
        return []

def search_ocr_on_elasticsearch_sync(keyword: str, limit: int = 500):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text"]}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
        return [{"filepath": hit['_source']['file_path'], "score": hit['_score'], "video_id": hit['_source']['video_id'], "shot_id": str(hit['_source']['shot_id']), "frame_id": hit['_source']['frame_id']} for hit in response["hits"]["hits"] if all(k in hit['_source'] for k in ['file_path', 'video_id', 'shot_id', 'frame_id'])]
    except NotFoundError: return []
    except Exception as e: print(f"Lỗi Elasticsearch OCR: {e}"); return []

def search_asr_on_elasticsearch_sync(keyword: str, limit: int = 500):
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
    except NotFoundError: return []
    except Exception as e: print(f"Lỗi Elasticsearch ASR: {e}"); return []

async def search_milvus_async(collection: Collection, collection_name: str, query_vectors: list, limit: int, expr: str = None):
    return await asyncio.to_thread(search_milvus_sync, collection, collection_name, query_vectors, limit, expr)

async def search_ocr_on_elasticsearch_async(keyword: str, limit: int = 500):
    return await asyncio.to_thread(search_ocr_on_elasticsearch_sync, keyword, limit)

async def search_asr_on_elasticsearch_async(keyword: str, limit: int = 500):
    return await asyncio.to_thread(search_asr_on_elasticsearch_sync, keyword, limit)

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

def package_response_with_urls(data: List[Dict[str, Any]], base_url: str):
    response_content = {"results": []}
    if not isinstance(data, list):
        return JSONResponse(content={"results": data})
    for item in data:
        if not isinstance(item, dict): continue
        def process_shot(shot_dict):
            if isinstance(shot_dict, dict) and 'filepath' in shot_dict:
                current_path = shot_dict.get('filepath')
                if current_path:
                    if current_path.startswith("/workspace"):
                        current_path = current_path.replace("/workspace", "/app", 1)
                    current_path = current_path.replace("_resized", "")
                    shot_dict['filepath'] = current_path
                    if 'url' not in shot_dict:
                         shot_dict['url'] = f"{base_url}images/{base64.urlsafe_b64encode(current_path.encode('utf-8')).decode('utf-8')}"
        if 'shots' in item and isinstance(item['shots'], list):
            for shot in item['shots']: process_shot(shot)
        if 'best_shot' in item: process_shot(item['best_shot'])
        if 'clusters' in item and isinstance(item['clusters'], list):
            for cluster in item['clusters']:
                 if isinstance(cluster, dict):
                    if 'shots' in cluster and isinstance(cluster['shots'], list):
                        for shot in cluster['shots']: process_shot(shot)
                    if 'best_shot' in cluster: process_shot(cluster['best_shot'])
    response_content["results"] = data
    return JSONResponse(content=response_content)

async def get_embeddings_for_query(
    client: httpx.AsyncClient, 
    text_queries: List[str], 
    image_content: Optional[bytes], 
    models: List[str], 
    query_image_info: Optional[Dict] = None,
    is_fusion: bool = False # <-- THÊM THAM SỐ NÀY
) -> Dict[str, List[List[float]]]:
    tasks = []
    model_url_map = {"beit3": BEIT3_WORKER_URL, "bge": BGE_WORKER_URL, "unite": UNITE_WORKER_URL}
    
    async def get_model_embedding(model_name: str) -> tuple[str, list]:
        url = model_url_map.get(model_name)
        if not url: return model_name, []
        
        files = None
        if image_content and query_image_info:
            files = {'image_file': (query_image_info['filename'], image_content, query_image_info['content_type'])}
        
        try:
            embeddings = []
            
            # [SỬA ĐỔI] Logic để xử lý fusion
            if is_fusion and image_content and text_queries:
                # Gửi cả text và ảnh đến worker (chỉ Unite hỗ trợ)
                data = {'text_query': text_queries[0]}
                resp = await client.post(url, files=files, data=data, timeout=30.0)
                if resp.status_code == 200:
                    embeddings.extend(resp.json().get('embedding', []))
            else:
                # Logic cũ: gửi từng query text hoặc chỉ ảnh
                queries = text_queries or [""] 
                for q in queries:
                    data = {'text_query': q} if q else {}
                    resp = await client.post(url, files=files, data=data, timeout=20.0)
                    if resp.status_code == 200:
                        embeddings.extend(resp.json().get('embedding', []))
            
            return model_name, embeddings
        except Exception as e:
            print(f"Error getting embedding for {model_name}: {e}")
            return model_name, []

    for model in models:
        if model in model_url_map:
            tasks.append(get_model_embedding(model))
    
    results = await asyncio.gather(*tasks)
    return {model_name: vecs for model_name, vecs in results}

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
    # Luồng xử lý: Luôn Translate -> (tùy chọn) Expand -> (tùy chọn) Enhance
    base_query = await translate_query(request_data.query)
    queries_to_process = expand_query_parallel(base_query) if request_data.expand else [base_query]
    if request_data.enhance:
        # Chạy enhance_query (hàm đồng bộ) trong một thread để không block event loop
        loop = asyncio.get_running_loop()
        final_queries = await loop.run_in_executor(
            None, lambda: [enhance_query(q) for q in queries_to_process]
        )
    else:
        final_queries = queries_to_process
    return {"processed_query": " ".join(final_queries)}

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    extension = Path(image.filename).suffix
    temp_filename = f"{uuid.uuid4()}{extension}"
    temp_filepath = TEMP_UPLOAD_DIR / temp_filename
    try:
        with temp_filepath.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    finally:
        image.file.close()
    return {"temp_image_name": temp_filename}

# --- DRES API Endpoints ---
@app.post("/dres/login")
async def dres_login(login_data: DRESLoginRequest):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{DRES_BASE_URL}/api/v2/login", json=login_data.dict())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"DRES login failed: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while contacting DRES: {e}")

@app.get("/dres/list_evaluations")
async def dres_list_evaluations(session: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{DRES_BASE_URL}/api/v2/client/evaluation/list", params={"session": session})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"DRES list evaluations failed: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while contacting DRES: {e}")

@app.post("/dres/submit")
async def dres_submit(submit_data: DRESSubmitRequest):
    try:
        filename = os.path.basename(submit_data.filepath)
        frame_match = re.search(r'_(\d{6})\.', filename)
        if not frame_match:
            raise ValueError("Could not extract 6-digit frame number from filepath.")
        
        frame_number = int(frame_match.group(1))
        time_ms = int((frame_number / VIDEO_FPS) * 1000)
        
        video_item_name = os.path.splitext(submit_data.video_id)[0]
        submission_body = {"answerSets": [{"answers": [{"mediaItemName": video_item_name, "start": time_ms, "end": time_ms}]}]}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DRES_BASE_URL}/api/v2/submit/{submit_data.evaluationId}",
                params={"session": submit_data.sessionId},
                json=submission_body
            )
            response.raise_for_status()
            return response.json()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"DRES submission failed: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while contacting DRES: {e}")

class ImageGenTextRequest(BaseModel):
    query: str

@app.post("/generate_image_from_text")
async def generate_image_from_text(request_data: ImageGenTextRequest):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    
    try:
        # Gọi đến service sinh ảnh
        response = requests.post(IMAGE_GEN_WORKER_URL, json={"query": request_data.query}, timeout=60.0)
        response.raise_for_status() # Ném lỗi nếu status code là 4xx hoặc 5xx

        image_content = response.content
        
        # Lưu ảnh vào thư mục tạm
        temp_filename = f"gen_{uuid.uuid4()}.png"
        temp_filepath = TEMP_UPLOAD_DIR / temp_filename
        with temp_filepath.open("wb") as f:
            f.write(image_content)

        # Tạo URL để UI có thể hiển thị
        encoded_path = base64.urlsafe_b64encode(str(temp_filepath).encode('utf-8')).decode('utf-8')
        image_url = f"/images/{encoded_path}"
        
        return {"temp_image_name": temp_filename, "image_url": image_url}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to contact image generation service: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image generation process: {e}")
    
# --- Search Endpoints ---
@app.post("/search")
async def search_unified(request: Request, search_data: str = Form(...), query_image: Optional[UploadFile] = File(None)):
    start_total_time = time.time()
    timings = {}

    try:
        search_data_model = UnifiedSearchRequest.parse_raw(search_data)
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid search data format: {e}")

    # --- Stage 1: OCR/ASR Filtering (Pre-filter) ---
    milvus_expr = None
    es_results_for_standalone_search = []
    is_filter_search = bool(search_data_model.ocr_query or search_data_model.asr_query)

    if is_filter_search:
        start_es = time.time()
        es_tasks = []
        if search_data_model.ocr_query:
            es_tasks.append(search_ocr_on_elasticsearch_async(search_data_model.ocr_query, limit=SEARCH_DEPTH * 5))
        if search_data_model.asr_query:
            es_tasks.append(search_asr_on_elasticsearch_async(search_data_model.asr_query, limit=SEARCH_DEPTH * 5))
        
        es_results_lists = await asyncio.gather(*es_tasks)
        es_res_map = {res['filepath']: res for res_list in es_results_lists for res in res_list}
        es_results_for_standalone_search = list(es_res_map.values())
        timings["ocr_asr_filtering_s"] = time.time() - start_es

        if not es_results_for_standalone_search:
            response_data = package_response_with_urls([], str(request.base_url))
            response_content = json.loads(response_data.body)
            response_content["processed_query"] = ""
            timings["total_request_s"] = time.time() - start_total_time
            response_content["timing_info"] = timings
            return JSONResponse(content=response_content)
        
        candidate_filepaths = []
        base_path = "/workspace/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/dataset_test/retrieval/webp_type"
        for res in es_results_for_standalone_search:
            stem = get_filename_stem(res['filepath'])
            if stem:
                candidate_filepaths.append(f"{base_path}/{stem}.webp")
        
        if candidate_filepaths:
            formatted_paths = [f'"{p}"' for p in candidate_filepaths]
            milvus_expr = f'filepath in [{",".join(formatted_paths)}]'

    # --- Stage 2: Prepare Vector Search Query ---
    start_query_proc = time.time()
    final_queries_to_embed = []
    processed_query_for_ui = ""
    image_content, query_image_info = None, None
    models_to_use = search_data_model.models
    is_image_search = bool(query_image or search_data_model.query_image_name)
    is_gen_image_search = bool(search_data_model.generated_image_name)
    loop = asyncio.get_running_loop()
    
    if is_gen_image_search: # <-- THÊM KHỐI LOGIC NÀY
        models_to_use = ["unite"] # Bắt buộc dùng Unite
        base_query = await translate_query(search_data_model.query_text)
        final_queries_to_embed = [base_query] # Chỉ dùng query gốc, không expand/enhance
        
        temp_filepath = TEMP_UPLOAD_DIR / search_data_model.generated_image_name
        if temp_filepath.is_file():
            image_content = temp_filepath.read_bytes()
            query_image_info = {"filename": search_data_model.generated_image_name, "content_type": "image/png"}
        else:
            print(f"WARNING: Generated image file not found: {temp_filepath}")

    elif is_image_search:
        models_to_use = ["bge"]
        if search_data_model.image_search_text:
            translated_image_text = await translate_query(search_data_model.image_search_text)
            final_queries_to_embed = [translated_image_text]
        if query_image: 
            image_content = await query_image.read()
            query_image_info = {"filename": query_image.filename, "content_type": query_image.content_type}
        elif search_data_model.query_image_name:
            temp_filepath = TEMP_UPLOAD_DIR / search_data_model.query_image_name
            if temp_filepath.is_file():
                image_content = temp_filepath.read_bytes()
                query_image_info = {"filename": search_data_model.query_image_name, "content_type": "image/jpeg"}
            else: 
                print(f"WARNING: Temporary image file not found: {temp_filepath}")
    elif search_data_model.query_text:
        base_query = await translate_query(search_data_model.query_text)
        queries_to_process = await loop.run_in_executor(None, expand_query_parallel, base_query) if search_data_model.expand else [base_query]
        if search_data_model.enhance:
            final_queries_to_embed = await loop.run_in_executor(None, lambda: [enhance_query(q) for q in queries_to_process])
        else:
            final_queries_to_embed = queries_to_process
        processed_query_for_ui = " ".join(final_queries_to_embed)
    timings["query_processing_s"] = time.time() - start_query_proc
    
    # --- Stage 3: Execute Vector Search ---
    is_primary_search = bool(final_queries_to_embed or image_content)
    final_fused_results = []

    if is_primary_search:
        if is_filter_search and not milvus_expr:
            final_fused_results = []
        else:
            start_embed = time.time()
            async with httpx.AsyncClient() as client:
                results_by_model = await get_embeddings_for_query(client, final_queries_to_embed, image_content, models_to_use, query_image_info, is_fusion=is_gen_image_search )
            timings["embedding_generation_s"] = time.time() - start_embed
            
            if any(results_by_model.values()):
                start_milvus = time.time()
                milvus_tasks = []
                if "beit3" in models_to_use and results_by_model.get("beit3"):
                    milvus_tasks.append(search_milvus_async(beit3_collection, BEIT3_COLLECTION_NAME, results_by_model.get("beit3", []), SEARCH_DEPTH, expr=milvus_expr))
                else:
                    milvus_tasks.append(asyncio.sleep(0, result=[])) # Trả về list rỗng nếu không dùng

                if "bge" in models_to_use and results_by_model.get("bge"):
                    milvus_tasks.append(search_milvus_async(bge_collection, BGE_COLLECTION_NAME, results_by_model.get("bge", []), SEARCH_DEPTH, expr=milvus_expr))
                else:
                    milvus_tasks.append(asyncio.sleep(0, result=[]))

                if "unite" in models_to_use and results_by_model.get("unite"):
                    milvus_tasks.append(search_milvus_async(unite_collection, UNITE_COLLECTION_NAME, results_by_model.get("unite", []), SEARCH_DEPTH, expr=milvus_expr))
                else:
                    milvus_tasks.append(asyncio.sleep(0, result=[]))
                
                beit3_res, bge_res, unite_res = await asyncio.gather(*milvus_tasks)
                timings["vector_search_s"] = time.time() - start_milvus
                
                start_post_proc = time.time()
                milvus_results_dict = {}
                if beit3_res: milvus_results_dict["beit3"] = beit3_res
                if bge_res: milvus_results_dict["bge"] = bge_res
                if unite_res: milvus_results_dict["unite"] = unite_res
                
                milvus_weights = {m: w for m, w in MODEL_WEIGHTS.items() if m in models_to_use}
                final_fused_results = reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res, "unite": unite_res}, milvus_weights)
                timings["post_processing_s"] = time.time() - start_post_proc
    
    elif is_filter_search:
        start_post_proc = time.time()
        for res in es_results_for_standalone_search:
            res['rrf_score'] = res.pop('score', 0.0)
        final_fused_results = sorted(es_results_for_standalone_search, key=lambda x: x.get('rrf_score', 0), reverse=True)
        timings["post_processing_s"] = time.time() - start_post_proc

    # --- Stage 4: Final Filtering and Response Packaging ---
    if final_fused_results:
        start_post_proc_2 = time.time()
        clustered_results = process_and_cluster_results_optimized(final_fused_results)
        final_results = clustered_results
        
        # --- TỐI ƯU ---
        # Chạy hàm lọc nặng (get_valid_filepaths_for_strict_search) trong một thread riêng
        # để không làm nghẽn event loop chính.
        if search_data_model.filters and clustered_results:
            all_filepaths = {s['filepath'] for c in clustered_results for s in c.get('shots', []) if 'filepath' in s}
            
            # Sử dụng asyncio.to_thread để chạy hàm đồng bộ trong background
            valid_filepaths = await asyncio.to_thread(
                get_valid_filepaths_for_strict_search, all_filepaths, search_data_model.filters
            )
            
            final_results = [
                c for c in clustered_results 
                if any(s['filepath'] in valid_filepaths for s in c.get('shots',[]))
            ]
        
        if "post_processing_s" in timings:
            timings["post_processing_s"] += time.time() - start_post_proc_2
        else:
            timings["post_processing_s"] = time.time() - start_post_proc_2
    else:
        final_results = []
    
    response_data = package_response_with_urls(final_results[:TOP_K_RESULTS], str(request.base_url))
    response_content = json.loads(response_data.body)
    response_content["processed_query"] = processed_query_for_ui

    timings["total_request_s"] = time.time() - start_total_time
    response_content["timing_info"] = timings 
    
    return JSONResponse(content=response_content)

@app.post("/temporal_search", response_class=JSONResponse)
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    start_total_time = time.time()
    timings = {}

    models, stages, filters = request_data.models, request_data.stages, request_data.filters
    if not stages or not models:
        raise HTTPException(status_code=400, detail="Stages and models are required.")
    
    processed_queries_for_ui = []

    # --- Helper function to get results for a single stage ---
    async def get_stage_results(client: httpx.AsyncClient, stage: StageData):
        # --- CHANGE: Updated logic to handle image and text queries ---
        has_vector_query = bool(stage.query or stage.query_image_name)
        has_ocr_asr_filter = bool(stage.ocr_query or stage.asr_query)
        milvus_expr = None
        all_es_results = []

        if has_ocr_asr_filter:
            es_tasks = []
            if stage.ocr_query:
                es_tasks.append(search_ocr_on_elasticsearch_async(stage.ocr_query, limit=SEARCH_DEPTH * 5))
            if stage.asr_query:
                es_tasks.append(search_asr_on_elasticsearch_async(stage.asr_query, limit=SEARCH_DEPTH * 5))
            
            es_results_lists = await asyncio.gather(*es_tasks)
            es_res_map = {res['filepath']: res for res_list in es_results_lists for res in res_list}
            all_es_results = list(es_res_map.values())
            
            if not all_es_results: return []

            candidate_filepaths = []
            base_path = "/workspace/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/dataset_test/retrieval/webp_type"
            for res in all_es_results:
                stem = get_filename_stem(res['filepath'])
                if stem: candidate_filepaths.append(f"{base_path}/{stem}.webp")
            
            if candidate_filepaths:
                formatted_paths = [f'"{p}"' for p in candidate_filepaths]
                milvus_expr = f'filepath in [{",".join(formatted_paths)}]'
            else:
                return []

        if has_vector_query:
            results_by_model = {}
            is_fusion_stage = bool(stage.generated_image_name and stage.query)
            if is_fusion_stage: # <-- THÊM KHỐI LOGIC NÀY
                # Logic cho stage fusion
                models_for_stage = ["unite"]
                base_query = await translate_query(stage.query)
                queries_to_embed = [base_query]
                processed_queries_for_ui.append(f"Gen-Image Fusion: {stage.query}")
                
                temp_filepath = TEMP_UPLOAD_DIR / stage.generated_image_name
                if not temp_filepath.is_file():
                    print(f"WARNING: Generated image for temporal stage not found: {temp_filepath}")
                    return []
                
                image_content = temp_filepath.read_bytes()
                query_image_info = {"filename": stage.generated_image_name, "content_type": "image/png"}
                
                results_by_model = await get_embeddings_for_query(
                    client, queries_to_embed, image_content, models_for_stage, query_image_info, is_fusion=True
                )
            # Check if the stage is an image search or a text search
            elif stage.query_image_name:
                # Logic for image stages
                temp_filepath = TEMP_UPLOAD_DIR / stage.query_image_name
                if not temp_filepath.is_file():
                    print(f"WARNING: Image file for temporal stage not found: {temp_filepath}")
                    return []
                
                image_content = temp_filepath.read_bytes()
                query_image_info = {"filename": stage.query_image_name, "content_type": "image/jpeg"}
                
                # Image search typically uses a specific model
                models_for_image = ["bge"] 
                results_by_model = await get_embeddings_for_query(client, [], image_content, models_for_image, query_image_info, is_fusion=False)
                processed_queries_for_ui.append(f"Image: {stage.query_image_name}")

            else: # Fallback to text search logic
                # This logic is for text queries (same as before)
                if asyncio.iscoroutinefunction(translate_query):
                    base_query = await translate_query(stage.query)
                else:
                    base_query = await asyncio.to_thread(translate_query, stage.query)

                queries_to_process = [base_query]
                if stage.expand:
                    queries_to_process = await asyncio.to_thread(expand_query_parallel, base_query)
                
                queries_to_embed = queries_to_process
                if stage.enhance:
                    queries_to_embed = await asyncio.to_thread(lambda: [enhance_query(q) for q in queries_to_process])
                
                processed_queries_for_ui.append(" ".join(queries_to_embed))
                results_by_model = await get_embeddings_for_query(client, queries_to_embed, None, models, is_fusion=False)
            
            # --- Common logic for both image and text vector search ---
            if has_ocr_asr_filter and not milvus_expr: return []

            if not any(results_by_model.values()): return []
            
            milvus_tasks = [
                search_milvus_async(beit3_collection, BEIT3_COLLECTION_NAME, results_by_model.get("beit3", []), SEARCH_DEPTH_PER_STAGE, expr=milvus_expr),
                search_milvus_async(bge_collection, BGE_COLLECTION_NAME, results_by_model.get("bge", []), SEARCH_DEPTH_PER_STAGE, expr=milvus_expr),
                search_milvus_async(unite_collection, UNITE_COLLECTION_NAME, results_by_model.get("unite", []), SEARCH_DEPTH_PER_STAGE, expr=milvus_expr)
            ]
            beit3_res, bge_res, unite_res = await asyncio.gather(*milvus_tasks)
            return reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res, "unite": unite_res}, MODEL_WEIGHTS)
        
        elif has_ocr_asr_filter:
            processed_queries_for_ui.append(f"OCR/ASR: {stage.ocr_query or ''} / {stage.asr_query or ''}")
            for res in all_es_results:
                res['rrf_score'] = res.pop('score', 0.0)
            return sorted(all_es_results, key=lambda x: x.get('rrf_score', 0), reverse=True)
        
        else:
            return []

    # --- Main Execution Flow ---
    start_stages = time.time()
    async with httpx.AsyncClient(timeout=120.0) as client:
        stage_tasks = [get_stage_results(client, stage) for stage in request_data.stages]
        all_stage_candidates = await asyncio.gather(*stage_tasks, return_exceptions=True)
    timings["stage_candidate_gathering_s"] = time.time() - start_stages

    valid_stage_results = [res for res in all_stage_candidates if isinstance(res, list)]
    
    clustered_results_by_stage = [process_and_cluster_results_optimized(res) for res in valid_stage_results]
    clustered_results_by_stage = [c for c in clustered_results_by_stage if c]

    def create_empty_response():
        response = package_response_with_urls([], str(request.base_url))
        content = json.loads(response.body)
        content["processed_queries"] = processed_queries_for_ui
        timings["total_request_s"] = time.time() - start_total_time
        content["timing_info"] = timings
        return JSONResponse(content=content)

    if len(clustered_results_by_stage) < len(stages):
        return create_empty_response()
    
    # --- Sequence Assembly using Recursive Backtracking ---
    start_assembly = time.time()
    
    for stage_clusters in clustered_results_by_stage:
        for cluster in stage_clusters:
            if cluster.get('shots'):
                shot_ids_int = [s['shot_id_int'] for s in cluster['shots'] if 'shot_id_int' in s]
                if shot_ids_int: 
                    cluster['min_shot_id'] = min(shot_ids_int)
                    cluster['max_shot_id'] = max(shot_ids_int)
                    cluster['video_id'] = cluster['best_shot']['video_id']

    clusters_by_video = defaultdict(lambda: defaultdict(list))
    for i, stage_clusters in enumerate(clustered_results_by_stage):
        for cluster in stage_clusters:
            if 'video_id' in cluster:
                clusters_by_video[cluster['video_id']][i].append(cluster)

    all_valid_sequences = []
    for video_id, video_stages in clusters_by_video.items():
        if len(video_stages) < len(stages):
            continue

        def find_sequences_recursive(stage_idx: int, current_sequence: list):
            if stage_idx == len(stages):
                all_valid_sequences.append(list(current_sequence))
                return

            for next_cluster in video_stages.get(stage_idx, []):
                if not current_sequence or next_cluster.get('min_shot_id', -1) > current_sequence[-1].get('max_shot_id', -1):
                    current_sequence.append(next_cluster)
                    find_sequences_recursive(stage_idx + 1, current_sequence)
                    current_sequence.pop()

        find_sequences_recursive(0, [])

    timings["sequence_assembly_s"] = time.time() - start_assembly
    
    if not all_valid_sequences: 
        return create_empty_response()
        
    # --- Final Processing and Filtering ---
    start_final_proc = time.time()
    processed_sequences = []
    for cluster_seq in all_valid_sequences:
        if not cluster_seq: continue
        avg_score = sum(c.get('cluster_score', 0) for c in cluster_seq) / len(cluster_seq)
        processed_sequences.append({
            "average_rrf_score": avg_score,
            "clusters": cluster_seq,
            "shots": [c['best_shot'] for c in cluster_seq],
            "video_id": cluster_seq[0].get('video_id', 'N/A')
        })
    
    sequences_to_filter = sorted(processed_sequences, key=lambda x: x['average_rrf_score'], reverse=True)
    
    if filters and (filters.counting or filters.positioning):
        filter_tasks = [
            asyncio.to_thread(is_temporal_sequence_valid, seq, filters) 
            for seq in sequences_to_filter
        ]
        filter_results = await asyncio.gather(*filter_tasks)
        
        final_sequences = [
            seq for seq, is_valid in zip(sequences_to_filter, filter_results) if is_valid
        ]
    else:
        final_sequences = sequences_to_filter
        
    timings["final_processing_s"] = time.time() - start_final_proc
    
    # --- Package and Return Response ---
    response = package_response_with_urls(final_sequences[:MAX_SEQUENCES_TO_RETURN], str(request.base_url))
    content = json.loads(response.body)
    content["processed_queries"] = processed_queries_for_ui
    timings["total_request_s"] = time.time() - start_total_time
    content["timing_info"] = timings
    
    return JSONResponse(content=content)

@app.post("/check_temporal_frames")
async def check_temporal_frames(request_data: CheckFramesRequest) -> List[str]:
    base_filepath = request_data.base_filepath
    if not base_filepath or not os.path.isfile(base_filepath):
        raise HTTPException(status_code=404, detail="Base filepath not found or does not exist.")
    try:
        def find_frames():
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
        return await asyncio.to_thread(find_frames)
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