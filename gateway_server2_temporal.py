import os
import sys
import time
import traceback
import base64
import asyncio
import operator
import json
import re  # Thêm import cho biểu thức chính quy
from collections import defaultdict
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ValidationError
from pymilvus import Collection, connections, utility
from elasticsearch import Elasticsearch, AsyncElasticsearch
import polars as pl
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
# app.mount("/", StaticFiles(directory=BASE_DIR), name="static")
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

# WHY: Centralizes collection names for pre-loading and makes the ES limit easy to configure.
ALL_MILVUS_COLLECTIONS = [BEIT3_COLLECTION, BGE_COLLECTION, UNITE_COLLECTION]
ES_CANDIDATE_LIMIT = 1000

# Trọng số cho các mô hình khi hợp nhất
MODEL_WEIGHTS = {"beit3": 0.4, "bge": 0.3, "unite": 0.3}

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

es: Optional[AsyncElasticsearch] = None # WHY: Type hint for the async client.
OBJECT_COUNTS_DF: Optional[pl.DataFrame] = None
OBJECT_POSITIONS_DF: Optional[pl.DataFrame] = None

# --- Pydantic Models ---
class ObjectCountFilter(BaseModel): conditions: Dict[str, str] = {}
class PositionBox(BaseModel): label: str; box: List[float]
class ObjectPositionFilter(BaseModel): boxes: List[PositionBox] = []
class ObjectFilters(BaseModel): counting: Optional[ObjectCountFilter] = None; positioning: Optional[ObjectPositionFilter] = None
class StageData(BaseModel):
    query: str
    expand: bool
    enhance: bool
    ocr_query: Optional[str] = None  # ADD THIS
    asr_query: Optional[str] = None  # ADD THIS
class TemporalSearchRequest(BaseModel):
    stages: list[StageData]
    models: List[str] = ["beit3", "bge", "unite"]
    cluster: bool = False  # Note: this parameter is unused in your current code
    filters: Optional[ObjectFilters] = None

class ProcessQueryRequest(BaseModel): query: str; enhance: bool; expand: bool

class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    # The new ocr_query field from the user request is already here in your provided code
    # But let's formalize it and add ASR
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None   # ADD THIS
    models: List[str] = ["beit3", "bge", "unite"]
    enhance: bool = False
    expand: bool = False
    filters: Optional[ObjectFilters] = None

# >>> BẮT ĐẦU MÃ MỚI: Pydantic Model cho request kiểm tra frame <<<
class CheckFramesRequest(BaseModel):
    base_filepath: str
# >>> KẾT THÚC MÃ MỚI <<<

@app.on_event("startup")
async def startup_event():
    global es, OBJECT_COUNTS_DF, OBJECT_POSITIONS_DF
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Milvus connection successful. ---")
        print("--- Pre-loading Milvus collections into memory... ---")
        # WHY: This loop loads all collections once, preventing slow loads on every search request.
        for collection_name in ALL_MILVUS_COLLECTIONS:
            if utility.has_collection(collection_name):
                start_time = time.time()
                Collection(collection_name).load()
                print(f"--- Collection '{collection_name}' loaded in {time.time() - start_time:.2f}s ---")
            else:
                print(f"!!! WARNING: Milvus collection '{collection_name}' not found. !!!")
    except Exception as e:
        print(f"FATAL: Could not connect to Milvus. Error: {e}")

    # Async Elasticsearch Client Initialization
    try:
        es = AsyncElasticsearch(hosts=[ELASTICSEARCH_HOST])
        await es.ping() # WHY: Must await calls on an async client.
        print("--- Async Elasticsearch connection successful. ---")
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
# ... (Tất cả các hàm helper của bạn giữ nguyên, không thay đổi) ...
async def get_candidate_filepaths_from_es(ocr_query: Optional[str], asr_query: Optional[str]) -> Optional[set[str]]:
    """
    Searches Elasticsearch for OCR/ASR queries and returns a set of candidate filepaths.
    Returns the INTERSECTION if both queries are provided.
    Returns None if no queries are given, indicating no pre-filtering is needed.
    """
    if not ocr_query and not asr_query:
        return None

    ocr_filepaths = set()
    if ocr_query:
        # Assuming search_ocr_on_elasticsearch returns a list of dicts with 'filepath'
        ocr_results = await search_ocr_on_elasticsearch(ocr_query, limit=5000) # Use a high limit
        ocr_filepaths = {res['filepath'] for res in ocr_results}
        # If OCR is the only filter and it returns no results, we can stop early.
        if not ocr_filepaths and not asr_query:
            return set()

    asr_filepaths = set()
    if asr_query:
        asr_results = await search_asr_on_elasticsearch(asr_query, limit=5000) # Use a high limit
        asr_filepaths = {res['filepath'] for res in asr_results}
        # If ASR is the only filter and it returns no results, we can stop early.
        if not asr_filepaths and not ocr_query:
            return set()

    if ocr_query and asr_query:
        return ocr_filepaths.intersection(asr_filepaths)
    elif ocr_query:
        #print(list(ocr_filepaths)[:5])
        return ocr_filepaths
    else: # asr_query must be true here
        return asr_filepaths

async def get_candidate_stems_from_es(ocr_query: Optional[str], asr_query: Optional[str]) -> Optional[set[str]]:
    """
    Searches Elasticsearch for OCR/ASR queries and returns a set of candidate FILENAME STEMS.
    Returns the INTERSECTION if both queries are provided.
    Returns None if no queries are given, indicating no pre-filtering is needed.
    """
    if not ocr_query and not asr_query:
        return None

    ocr_stems = set()
    if ocr_query:
        # WHY: Calls the async function correctly without the 'limit' argument.
        ocr_results = await search_ocr_on_elasticsearch(ocr_query)
        # WHY: This is the key change. It extracts the base filename without extension.
        # e.g., "/path/to/image.png" becomes "image"
        ocr_stems = {os.path.splitext(os.path.basename(res['filepath']))[0] for res in ocr_results}
        if not ocr_stems and not asr_query:
            return set()

    asr_stems = set()
    if asr_query:
        asr_results = await search_asr_on_elasticsearch(asr_query)
        asr_stems = {os.path.splitext(os.path.basename(res['filepath']))[0] for res in asr_results}
        if not asr_stems and not ocr_query:
            return set()

    if ocr_query and asr_query:
        return ocr_stems.intersection(asr_stems)
    elif ocr_query:
        return ocr_stems
    else: # asr_query must be true here
        return asr_stems

def remap_filepaths_in_results(data: List[Dict[str, Any]]):
    """
    Recursively remaps 'filepath' fields from /workspace to /app in place.
    This ensures the frontend always receives consistent, production-ready paths.
    """
    if not isinstance(data, list):
        return data

    for item in data:
        if not isinstance(item, dict):
            continue

        # Inner recursive function to process a single shot/dictionary
        def process_shot(shot_dict):
            if isinstance(shot_dict, dict) and 'filepath' in shot_dict:
                original_path = shot_dict.get('filepath')
                if original_path and original_path.startswith("/workspace"):
                    shot_dict['filepath'] = original_path.replace("/workspace", "/app", 1)
        
        # Process nested shots
        if 'shots' in item and isinstance(item['shots'], list):
            for shot in item['shots']:
                process_shot(shot)
        
        # Process best_shot
        if 'best_shot' in item:
            process_shot(item['best_shot'])

        # Process nested clusters (for temporal search)
        if 'clusters' in item and isinstance(item['clusters'], list):
            for cluster in item['clusters']:
                 if isinstance(cluster, dict):
                    if 'shots' in cluster and isinstance(cluster['shots'], list):
                        for shot in cluster['shots']:
                            process_shot(shot)
                    if 'best_shot' in cluster:
                        process_shot(cluster['best_shot'])
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
    candidate_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_filepaths}
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
    return {fp for fp in all_filepaths if os.path.splitext(os.path.basename(fp))[0] in valid_stems}

def search_milvus(collection_name: str, query_vectors: list, limit: int, expr: str = None):
    try:
        if not utility.has_collection(collection_name) or not len(query_vectors): return []
        collection = Collection(collection_name)
        
        #collection.load()

        index_type = COLLECTION_TO_INDEX_TYPE.get(collection_name, "DEFAULT")
        search_params = SEARCH_PARAMS.get(index_type)
        if not search_params: search_params = SEARCH_PARAMS["DEFAULT"]
        anns_field = "image_embedding"
        output_fields = ["filepath", "video_id", "shot_id", "frame_id"]
        results = collection.search(query_vectors, anns_field, search_params, limit=limit, output_fields=output_fields, expr=expr)
        parsed_results = []
        for s in results:
            for h in s:
                res = {
                    "filepath": h.entity.get("filepath"), "score": h.distance,
                    "video_id": h.entity.get("video_id"), "frame_id": h.entity.get("frame_id"),
                    "shot_id": str(h.entity.get("shot_id")), "caption_text": h.entity.get("caption_text")
                }
                parsed_results.append(res)
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

async def search_ocr_on_elasticsearch(keyword: str, limit: int=100):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text"]}}}
    try:
        response = await es.search(index=OCR_ASR_INDEX_NAME, body=query, size=ES_CANDIDATE_LIMIT)
    except Exception as e:
        print(f"Lỗi Elasticsearch: {e}")
        return []
        
    results = []
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        if all(k in source for k in ['file_path', 'video_id', 'shot_id', 'frame_id']):
            results.append({
                "filepath": source['file_path'],
                "score": hit['_score'],
                "video_id": source['video_id'],
                "shot_id": source['shot_id'],
                "frame_id": source['frame_id']
            })
    return results


async def search_asr_on_elasticsearch(keyword: str, limit: int=100):
    if not es: 
        return []
    
    query = {
        "query": {
            "multi_match": {
                "query": keyword,
                "fields": [
                    "asr_text^3",           
                    "text^1",               
                    "_all"                  
                ],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        },
        "highlight": {
            "fields": {
                "asr_text": {},
                "text": {}
            }
        }
    }
    
    try:
        response = await es.search(index=OCR_ASR_INDEX_NAME, body=query, size=ES_CANDIDATE_LIMIT)
        total_hits = response["hits"]["total"]["value"]
        
        if total_hits > 0:
            results = []
            for hit in response["hits"]["hits"]:
                source = hit['_source']
                
                filepath = source.get('file_path') or source.get('filepath') or source.get('image_path')
                video_id = source.get('video_id') or source.get('video')
                shot_id = source.get('shot_id') or source.get('shot')
                frame_id = source.get('frame_id') or source.get('frame')
                
                if filepath and video_id and shot_id is not None:
                    result = {
                        "filepath": filepath,
                        "score": hit['_score'],
                        "video_id": video_id,
                        "shot_id": str(shot_id),
                        "frame_id": frame_id
                    }
                    results.append(result)
            
            return results
        else:
            return []
            
    except Exception as e:
        print(f"Error in ASR search: {e}")
        return []

def process_and_cluster_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results: return []
    shots_by_video = defaultdict(list)
    for res in results:
        if not all(k in res for k in ['video_id', 'shot_id']): continue
        try: shot_id_str = str(res['shot_id']); res['shot_id_int'] = int(shot_id_str); shots_by_video[res['video_id']].append(res)
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
    if not isinstance(data, list):
        return data

    for item in data:
        if not isinstance(item, dict):
            continue
            
        # Process shots in the main item
        if 'shots' in item and isinstance(item['shots'], list):
            for shot in item['shots']:
                if isinstance(shot, dict) and shot.get('filepath') and 'url' not in shot:
                    shot['url'] = f"{base_url}images/{base64.urlsafe_b64encode(shot['filepath'].encode('utf-8')).decode('utf-8')}"

        # Process the best_shot in the main item
        if 'best_shot' in item and isinstance(item['best_shot'], dict):
            best_shot = item['best_shot']
            if best_shot.get('filepath') and 'url' not in best_shot:
                best_shot['url'] = f"{base_url}images/{base64.urlsafe_b64encode(best_shot['filepath'].encode('utf-8')).decode('utf-8')}"
        
        # Process clusters within the main item
        if 'clusters' in item and isinstance(item['clusters'], list):
            for cluster in item['clusters']:
                 if isinstance(cluster, dict):
                    # Process shots within a cluster
                    if 'shots' in cluster and isinstance(cluster['shots'], list):
                        for shot in cluster['shots']:
                             if isinstance(shot, dict) and shot.get('filepath') and 'url' not in shot:
                                shot['url'] = f"{base_url}images/{base64.urlsafe_b64encode(shot['filepath'].encode('utf-8')).decode('utf-8')}"
                    # Process best_shot within a cluster
                    if 'best_shot' in cluster and isinstance(cluster['best_shot'], dict):
                        best_shot_in_cluster = cluster['best_shot']
                        if best_shot_in_cluster.get('filepath') and 'url' not in best_shot_in_cluster:
                           best_shot_in_cluster['url'] = f"{base_url}images/{base64.urlsafe_b64encode(best_shot_in_cluster['filepath'].encode('utf-8')).decode('utf-8')}"

    return data


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = os.path.join(BASE_DIR, "ui1_2_temporal.html") # Đổi tên file UI nếu cần
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
    
    # Check for a valid query
    has_vector_query = bool(search_data_model.query_text or query_image)
    has_text_filter = bool(search_data_model.ocr_query or search_data_model.asr_query)
    
    if not has_vector_query and not has_text_filter:
        raise HTTPException(status_code=400, detail="A text query, image query, or OCR/ASR filter query is required.")
    
    if not search_data_model.models:
        raise HTTPException(status_code=400, detail="At least one model must be selected.")
    
    # --- Step 1: Pre-filtering with Elasticsearch (if requested) ---
    # candidate_filepaths = await get_candidate_filepaths_from_es(
    #     search_data_model.ocr_query,
    #     search_data_model.asr_query
    # )
    
    # milvus_expr = None
    # if candidate_filepaths is not None:
    #     if not candidate_filepaths: # If filter returns zero results, we are done.
    #         return []
    #     # WARNING: This can fail if candidate_filepaths is too large.
    #     # A production system would need to chunk this.
    #     milvus_expr = f"filepath in {list(candidate_filepaths)}"
    candidate_stems = await get_candidate_stems_from_es(
        search_data_model.ocr_query,
        search_data_model.asr_query
    )

    milvus_expr = None
    if candidate_stems is not None:
        if not candidate_stems: # If filter returns zero results, we are done.
            return []
        # WHY: This creates a query like '(filepath like "%stem1%" or filepath like "%stem2%")'
        # This allows Milvus to match the stem against its full filepaths, solving the mismatch.
        # It is slower than an exact 'in' match, but works with your existing data.
        expr_parts = [f'filepath like "%{stem}%"' for stem in candidate_stems]
        milvus_expr = " or ".join(expr_parts)
        
    fused_results = []

    # --- Step 2: Perform Vector Search OR Use ES results directly ---
    if has_vector_query:
        # This block is mostly the same as your old /search endpoint
        if not search_data_model.models:
            raise HTTPException(status_code=400, detail="At least one model must be selected for vector search.")
        final_queries_to_embed = []
        image_content = await query_image.read() if query_image else None
        
        if search_data_model.query_text:
            base_query = translate_query(search_data_model.query_text)
            queries_to_process = expand_query_parallel(base_query) if search_data_model.expand else [base_query]
            final_queries_to_embed = [enhance_query(q) for q in queries_to_process] if search_data_model.enhance else queries_to_process

    

        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks, model_order = [], []
            model_url_map = {"beit3": BEIT3_WORKER_URL, "bge": BGE_WORKER_URL, "unite": UNITE_WORKER_URL}

            for model in search_data_model.models:
                url = model_url_map.get(model)
                if not url: continue
                
                can_process_image = image_content
                files = {'image_file': (query_image.filename, image_content, query_image.content_type)} if can_process_image else None
                
                if final_queries_to_embed:
                    for q in final_queries_to_embed:
                        tasks.append(client.post(url, files=files, data={'text_query': q}))
                        model_order.append(model)
                elif can_process_image:
                    tasks.append(client.post(url, files=files))
                    model_order.append(model)
            
            if not tasks: raise HTTPException(status_code=400, detail="No valid query to process.")
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            results_by_model = defaultdict(list)
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception): print(f"Error calling model {model_order[i]}: {resp}"); continue
                if resp.status_code == 200: results_by_model[model_order[i]].extend(resp.json()['embedding'])
                else: print(f"Error from model {model_order[i]} (status {resp.status_code}): {resp.text}")
        
        beit3_res = search_milvus(BEIT3_COLLECTION, results_by_model.get("beit3", []), SEARCH_DEPTH, expr=milvus_expr)
        bge_res = search_milvus(BGE_COLLECTION, results_by_model.get("bge", []), SEARCH_DEPTH, expr=milvus_expr)
        unite_res = search_milvus(UNITE_COLLECTION, results_by_model.get("unite", []), SEARCH_DEPTH, expr=milvus_expr)
        
        fused_results = reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res, "unite": unite_res}, MODEL_WEIGHTS)

    elif has_text_filter:
        # Scenario: Only OCR/ASR filter, no vector search.
        # The candidates *are* the results. We just need to format them.
        # A simple way is to re-fetch from ES to get all metadata.
        if search_data_model.ocr_query:
            fused_results.extend(await search_ocr_on_elasticsearch(search_data_model.ocr_query, limit=TOP_K_RESULTS*2))
        if search_data_model.asr_query:
            fused_results.extend(await search_asr_on_elasticsearch(search_data_model.asr_query, limit=TOP_K_RESULTS*2))
        
        # Deduplicate and format
        seen_paths = set()
        deduped_results = []
        for res in fused_results:
            if res['filepath'] not in seen_paths:
                res['rrf_score'] = res.pop('score', 0.0) # Standardize score key
                deduped_results.append(res)
                seen_paths.add(res['filepath'])
        fused_results = sorted(deduped_results, key=lambda x: x['rrf_score'], reverse=True)

    # --- Step 3: Clustering and Final Object Filtering (applies to all scenarios) ---
    clustered_results = process_and_cluster_results(fused_results)


    if search_data_model.filters:
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
    else:
        final_results = clustered_results
    
    final_results = remap_filepaths_in_results(final_results)
    #print(list(final_results)[:5])
    return add_image_urls(final_results[:TOP_K_RESULTS], str(request.base_url))

@app.post("/temporal_search")
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    models, stages, filters = request_data.models, request_data.stages, request_data.filters
    if not stages: raise HTTPException(status_code=400, detail="No stages provided.")
    if not models: raise HTTPException(status_code=400, detail="No models selected.")

    async def get_stage_results(client, stage: StageData):
        # Step 1: Get candidates from OCR/ASR for THIS stage
        # candidate_filepaths = await get_candidate_filepaths_from_es(stage.ocr_query, stage.asr_query)

        # milvus_expr = None
        # if candidate_filepaths is not None:
        #     if not candidate_filepaths: # No candidates for this stage, so it cannot be part of a sequence
        #         return [] 
        #     milvus_expr = f"filepath in {list(candidate_filepaths)}"
        candidate_stems = await get_candidate_stems_from_es(stage.ocr_query, stage.asr_query)

        milvus_expr = None
        if candidate_stems is not None:
            if not candidate_stems: # No candidates for this stage, so it cannot be part of a sequence.
                return []
            expr_parts = [f'filepath like "%{stem}%"' for stem in candidate_stems]
            milvus_expr = " or ".join(expr_parts)
        # Step 2: Perform vector search on the filtered candidates
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

        # Step 3: Pass the expr to all Milvus calls
        beit3_res = search_milvus(BEIT3_COLLECTION, vecs_by_model.get("beit3", []), SEARCH_DEPTH_PER_STAGE, expr=milvus_expr)
        bge_res = search_milvus(BGE_COLLECTION, vecs_by_model.get("bge", []), SEARCH_DEPTH_PER_STAGE, expr=milvus_expr)
        unite_res = search_milvus(UNITE_COLLECTION, vecs_by_model.get("unite", []), SEARCH_DEPTH_PER_STAGE, expr=milvus_expr)
        
        return reciprocal_rank_fusion({"beit3": beit3_res, "bge": bge_res, "unite": unite_res}, MODEL_WEIGHTS)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        all_stage_candidates = await asyncio.gather(*[get_stage_results(client, stage) for stage in stages])

    clustered_results_by_stage = [process_and_cluster_results(res) for res in all_stage_candidates]
    
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

    all_valid_cluster_sequences = []
    for video_id, video_stages in clusters_by_video.items():
        if len(video_stages) < len(stages): continue 
        
        def find_cluster_combinations(current_sequence, stage_idx):
            if stage_idx == len(stages):
                all_valid_cluster_sequences.append(list(current_sequence))
                return
            
            for next_cluster in video_stages.get(stage_idx, []):
                if not current_sequence or next_cluster.get('min_shot_id', -1) > current_sequence[-1].get('max_shot_id', -1):
                    current_sequence.append(next_cluster)
                    find_cluster_combinations(current_sequence, stage_idx + 1)
                    current_sequence.pop()

        find_cluster_combinations([], 0)

    if not all_valid_cluster_sequences: return []

    processed_sequences = []
    for cluster_seq in all_valid_cluster_sequences:
        if not cluster_seq: continue
        shot_sequence = [c['best_shot'] for c in cluster_seq]
        avg_score = sum(c.get('cluster_score', 0) for c in cluster_seq) / len(cluster_seq)
        video_id = cluster_seq[0].get('video_id', 'N/A')
        processed_sequences.append({
            "average_rrf_score": avg_score, 
            "clusters": cluster_seq, 
            "shots": shot_sequence, 
            "video_id": video_id
        })
    
    sequences_to_filter = sorted(processed_sequences, key=lambda x: x['average_rrf_score'], reverse=True)
    
    if filters: 
        filtered_sequences = [seq for seq in sequences_to_filter if is_temporal_sequence_valid(seq, filters)]
    else: 
        filtered_sequences = sequences_to_filter
        
    final_sequences = remap_filepaths_in_results(filtered_sequences)
    return add_image_urls(final_sequences[:MAX_SEQUENCES_TO_RETURN], str(request.base_url))


# >>> BẮT ĐẦU MÃ MỚI: API Endpoint để kiểm tra các frame tồn tại <<<
@app.post("/check_temporal_frames")
async def check_temporal_frames(request_data: CheckFramesRequest) -> List[str]:
    """
    Nhận một đường dẫn tệp cơ sở. Quét toàn bộ thư mục, tìm tất cả các
    frame thuộc cùng một VIDEO, sắp xếp chúng theo thứ tự, và trả về 10
    frame kề trước và 10 frame kề sau có tồn tại.
    """
    base_filepath = request_data.base_filepath
    if not base_filepath or not os.path.isfile(base_filepath):
        raise HTTPException(status_code=404, detail="Base filepath not found or does not exist.")

    try:
        # 1. Lấy thư mục và tên tệp mục tiêu
        directory = os.path.dirname(base_filepath)
        target_filename = os.path.basename(base_filepath)

        # 2. Trích xuất VIDEO_ID (ví dụ: 'L08_V022') từ tên tệp
        #    Regex này tìm mẫu Lxx_Vxxx ở đầu tên tệp
        video_match = re.match(r'^(L\d+_V\d+)', target_filename)
        if not video_match:
            # Nếu tên tệp không có định dạng video, chỉ trả về chính nó
            return [base_filepath]

        video_prefix = video_match.group(1)  # Sẽ là 'L08_V022'

        # 3. Quét tất cả các tệp trong thư mục và lọc những tệp thuộc cùng VIDEO
        all_frames_in_video = []
        all_files_in_dir = os.listdir(directory)
        for filename in all_files_in_dir:
            # Chỉ xử lý các tệp bắt đầu bằng video_prefix
            if filename.startswith(video_prefix):
                # Trích xuất số frame (con số cuối cùng trong tên tệp) để sắp xếp
                frame_num_match = re.search(r'_(\d+)\.[^.]+$', filename)
                if frame_num_match:
                    frame_num = int(frame_num_match.group(1))
                    all_frames_in_video.append({
                        'num': frame_num,
                        'path': os.path.join(directory, filename)
                    })

        # 4. Sắp xếp tất cả các frame của video theo số frame
        all_frames_in_video.sort(key=lambda x: x['num'])
        
        # Tạo danh sách chỉ chứa các đường dẫn đã được sắp xếp
        sorted_paths = [frame['path'] for frame in all_frames_in_video]

        # 5. Tìm vị trí (index) của tệp được click trong danh sách đã sắp xếp
        try:
            target_index = sorted_paths.index(base_filepath)
        except ValueError:
            return [base_filepath] # An toàn: nếu không tìm thấy, trả về chính nó

        # 6. Cắt danh sách để lấy 10 tệp trước và 10 tệp sau từ vị trí đó
        start_index = max(0, target_index - 10)
        end_index = min(len(sorted_paths), target_index + 11)

        # 7. Lấy ra các đường dẫn kết quả và trả về cho frontend
        result_files = sorted_paths[start_index:end_index]
        return result_files

    except Exception as e:
        print(f"ERROR in check_temporal_frames (final logic): {e}")
        traceback.print_exc()
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