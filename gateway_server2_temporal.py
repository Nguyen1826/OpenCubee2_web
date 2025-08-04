# FILE: gateway_server.py (Updated with ASR search functionality)

import os
import sys
import traceback
import base64
import asyncio
from collections import defaultdict
import httpx
import re
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pymilvus import Collection, connections, utility
from elasticsearch import Elasticsearch

# --- Path setup remains the same ---
COMMON_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)

# --- Import and debug block remains the same ---
try:
    from Cubi.translate_query import translate_text
    print("--- Gateway Server: Successfully imported 'translate_text' from Cubi subfolder. ---")
except ImportError as e:
    print("\n" + "="*80)
    print("!!! FATAL IMPORT ERROR: Could not import 'translate_text'. !!!")
    traceback.print_exc()
    print("="*80 + "\n")


# --- Configuration ---
BEIT3_WORKER_URL = "http://model-workers:8001/embed"
OPENCLIP_WORKER_URL = "http://model-workers:8002/embed"
ELASTICSEARCH_HOST = "http://elasticsearch2:9200"

OCR_INDEX_NAME = "opencubee_2" 
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
BEIT3_COLLECTION = "beit3_large_embeddings"
OPENCLIP_COLLECTION = "openclip_h14_embeddings"
SEARCH_DEPTH = 500
TOP_K_RESULTS = 50
ALLOWED_BASE_DIR = COMMON_PARENT_DIR
MAX_SEQUENCES_TO_RETURN = 20
INITIAL_CANDIDATES = 20
SEARCH_DEPTH_PER_STAGE = 200

es = None

# --- FastAPI App Initialization & Startup ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global es
    print("--- Gateway Server: Connecting to Milvus... ---")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Gateway Server: Milvus connection successful. ---")
    except Exception as e:
        print(f"FATAL: Could not connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}. Error: {e}")
    
    print("--- Gateway Server: Connecting to Elasticsearch... ---")
    try:
        es = Elasticsearch(ELASTICSEARCH_HOST)
        if not es.ping():
            raise ConnectionError("Elasticsearch ping failed.")
        print("--- Gateway Server: Elasticsearch connection successful. ---")
    except Exception as e:
        print(f"FATAL: Could not connect to Elasticsearch at {ELASTICSEARCH_HOST}. Error: {e}")
        es = None

# --- Helper Functions (Unchanged) ---
def extract_video_id(filepath: str) -> str:
    match = re.search(r'.*/(L\d+_V\d+)_', filepath)
    if match:
        return match.group(1)
    return None

def search_milvus(collection_name: str, query_vector, limit: int, min_filepath: str = None):
    try:
        if not utility.has_collection(collection_name): return []
        collection = Collection(collection_name)
        collection.load()
        expr = f"filepath > '{min_filepath}'" if min_filepath else None
        if expr:
            print(f"    - Milvus Query on '{collection_name}' with filter: {expr}")
        else:
            print(f"    - Milvus Query on '{collection_name}' with no filter.")
        search_params = {"metric_type": "COSINE", "params": {"ef": max(256, limit + 50)}}
        results = collection.search(query_vector, "embedding", search_params, limit=limit, output_fields=["filepath"], expr=expr)
        return [{"filepath": hit.entity.get("filepath"), "score": hit.distance} for hit in results[0]] if results else []
    except Exception as e:
        print(f"ERROR during Milvus search on '{collection_name}': {e}")
        traceback.print_exc()
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

def search_ocr_on_elasticsearch(keyword: str, limit: int=100):
    if not es:
        print("Error: Cannot connect to ElasticSearch")
        return []
    
    query = {
        "query": {
            "multi_match": {
                "query": keyword,
                "fields": ["ocr_text", "asr_text"]
            }
        }
    }

    try:
        response = es.search(index=OCR_INDEX_NAME, body=query)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return [] 
        
    results = []
    base_image_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/Enn/dataset/data_frame_ocr_png"
    
    for hit in response["hits"]["hits"]:
        source = hit['_source']
        video = source.get('video')
        shot_id = source.get('shot_id')
        frame_id = source.get('frame')

        if not all([video, shot_id, frame_id]):
            continue

        formatted_frame_id = str(frame_id).zfill(6)
        filename = f"{video}_{shot_id}_{formatted_frame_id}.png"
        filepath = os.path.join(base_image_path, filename)

        results.append({
            "filepath": filepath,
            "score": hit['_score'],
            "ocr_text": source.get('ocr_text', 'N/A'),
            "asr_text": source.get('asr_text', 'N/A'),
            "video": video,
            "shot_id": shot_id,
        })
        
    return results

# NEW: ASR-specific search function using the new dataset structure
def search_asr_from_json_metadata(keyword: str, limit: int=100):
    """Search specifically in ASR metadata files from the new dataset structure"""
    print(f"Searching ASR metadata for keyword: '{keyword}'")
    
    # Path to the ASR dataset with metadata files
    asr_dataset_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/ThoLe/image_with_asr"
    combined_metadata_file = os.path.join(asr_dataset_path, "all_asr_matching_metadata.json")
    
    results = []
    
    try:
        # Load the combined metadata file
        if not os.path.exists(combined_metadata_file):
            print(f"ASR metadata file not found: {combined_metadata_file}")
            return []
            
        with open(combined_metadata_file, 'r', encoding='utf-8') as f:
            asr_metadata = json.load(f)
        
        print(f"Loaded {len(asr_metadata)} ASR entries from metadata")
        
        # Search through ASR transcripts
        keyword_lower = keyword.lower()
        for entry in asr_metadata:
            asr_transcript = entry.get('asr_transcript', '').lower()
            
            # Simple text matching (you can enhance this with fuzzy matching if needed)
            if keyword_lower in asr_transcript:
                # Calculate a simple relevance score based on keyword frequency
                score = asr_transcript.count(keyword_lower) / len(asr_transcript.split()) * 100
                
                results.append({
                    "filepath": entry['output_path'],
                    "score": score,
                    "asr_text": entry['asr_transcript'],
                    "video": entry['video_name'],
                    "frame_timestamp_ms": entry['frame_timestamp_ms'],
                    "asr_start_ms": entry['asr_start_ms'],
                    "asr_end_ms": entry['asr_end_ms'],
                    "language": entry.get('language', 'unknown'),
                    "original_filename": entry['original_filename']
                })
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit results
        results = results[:limit]
        
        print(f"Found {len(results)} ASR matches for '{keyword}'")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing ASR metadata JSON: {e}")
    except Exception as e:
        print(f"Error searching ASR metadata: {e}")
    
    return results

# Alternative: Search using individual video metadata files
def search_asr_from_individual_metadata(keyword: str, limit: int=100):
    """Search ASR using individual video metadata files"""
    print(f"Searching individual ASR metadata files for keyword: '{keyword}'")
    
    # Chưa tạo
    asr_dataset_path = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/ThoLe/image_with_asr"
    
    results = []
    keyword_lower = keyword.lower()
    
    try:
        # Find all individual metadata files
        for metadata_file in Path(asr_dataset_path).glob("*_asr_matching_metadata.json"):
            print(f"Processing metadata file: {metadata_file}")
            
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    video_metadata = json.load(f)
                
                # Search through this video's ASR data
                for entry in video_metadata:
                    asr_transcript = entry.get('asr_transcript', '').lower()
                    
                    if keyword_lower in asr_transcript:
                        score = asr_transcript.count(keyword_lower) / len(asr_transcript.split()) * 100
                        
                        results.append({
                            "filepath": entry['output_path'],
                            "score": score,
                            "asr_text": entry['asr_transcript'],
                            "video": entry['video_name'],
                            "frame_timestamp_ms": entry['frame_timestamp_ms'],
                            "asr_start_ms": entry['asr_start_ms'],
                            "asr_end_ms": entry['asr_end_ms'],
                            "language": entry.get('language', 'unknown'),
                            "original_filename": entry['original_filename']
                        })
                        
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")
                continue
    
    except Exception as e:
        print(f"Error searching ASR metadata files: {e}")
    
    # Sort and limit results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:limit]

# Main ASR search function (uses the new dataset)
def search_asr_on_elasticsearch(keyword: str, limit: int=100):
    print("Using new ASR dataset structure for search")
    
    # Try combined metadata first, fall back to individual files
    results = search_asr_from_json_metadata(keyword, limit)
    
    if not results:
        print("No results from combined metadata, trying individual files...")
        results = search_asr_from_individual_metadata(keyword, limit)
    
    return results

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = os.path.join(ALLOWED_BASE_DIR, "Cubi", "ui", "ui1_2_temporal.html")
    if not os.path.exists(ui_path): raise HTTPException(status_code=500, detail=f"Error: ui1.html not found at expected path: {ui_path}")
    with open(ui_path, "r") as f: return HTMLResponse(content=f.read())

class TemporalSearchRequest(BaseModel):
    queries: list[str]

@app.post("/temporal_search")
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    queries = request_data.queries
    if not queries:
        raise HTTPException(status_code=400, detail="No queries provided.")

    print("\n" + "="*50)
    print("--- INITIATING MULTI-SEQUENCE TEMPORAL SEARCH ---")
    print(f"--- Received {len(queries)} stages. Finding up to {INITIAL_CANDIDATES} initial paths. ---")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # --- Stage 1: Get initial candidates ---
        print(f"\n--- Processing Stage 1: '{queries[0]}' to find starting points ---")
        translated_query = translate_text(queries[0])
        vec_responses = await asyncio.gather(
            client.post(BEIT3_WORKER_URL, data={'text_query': translated_query}),
            client.post(OPENCLIP_WORKER_URL, data={'text_query': translated_query})
        )
        vec_beit3 = vec_responses[0].json()['embedding']
        vec_opc = vec_responses[1].json()['embedding']
        beit3_results = search_milvus(BEIT3_COLLECTION, vec_beit3, SEARCH_DEPTH_PER_STAGE)
        openclip_results = search_milvus(OPENCLIP_COLLECTION, vec_opc, SEARCH_DEPTH_PER_STAGE)
        initial_candidates = reciprocal_rank_fusion({"beit3": beit3_results, "openclip": openclip_results}, {"beit3": 30, "openclip": 0.30})[:INITIAL_CANDIDATES]

        if not initial_candidates:
            print("--- !!! SEARCH FAILED: No candidates found for the first stage. !!! ---")
            return []
        print(f"--- Found {len(initial_candidates)} potential starting shots. ---")

        # --- Subsequent Stages with CORRECTED filtering ---
        all_valid_sequences = []
        for candidate in initial_candidates:
            video_id = extract_video_id(candidate['filepath'])
            if not video_id:
                continue

            print(f"\n  -> Branching from candidate: '{candidate['filepath']}' (Video: {video_id})")
            
            current_sequence = [candidate]
            last_successful_path = candidate['filepath']
            is_complete = True

            for i, query in enumerate(queries[1:]):
                stage_num = i + 2
                print(f"    -- Searching for Stage {stage_num}: '{query}' in Video {video_id} --")
                
                t_query = translate_text(query)
                vecs = await asyncio.gather(
                    client.post(BEIT3_WORKER_URL, data={'text_query': t_query}),
                    client.post(OPENCLIP_WORKER_URL, data={'text_query': t_query})
                )
                b_vec, o_vec = vecs[0].json()['embedding'], vecs[1].json()['embedding']
                
                b_res = search_milvus(BEIT3_COLLECTION, b_vec, SEARCH_DEPTH_PER_STAGE, min_filepath=last_successful_path)
                o_res = search_milvus(OPENCLIP_COLLECTION, o_vec, SEARCH_DEPTH_PER_STAGE, min_filepath=last_successful_path)
                
                fused_results = reciprocal_rank_fusion({"beit3": b_res, "openclip": o_res}, {"beit3": 30, "openclip": 0.30})
                filtered_by_video = [res for res in fused_results if extract_video_id(res['filepath']) == video_id]
                
                if not filtered_by_video:
                    print(f"    -- !!! Path broken. No result found for Stage {stage_num} in Video {video_id}. !!! --")
                    is_complete = False
                    break
                
                top_result = filtered_by_video[0]
                current_sequence.append(top_result)
                last_successful_path = top_result['filepath']
                print(f"    -- Found Stage {stage_num} match: '{last_successful_path}' --")

            if is_complete:
                print(f"  -> SUCCESS: Found a complete sequence of {len(current_sequence)} shots for Video {video_id}.")
                all_valid_sequences.append(current_sequence)
    
    # --- THIS IS THE CORRECTED FINALIZATION BLOCK ---
    print(f"\n--- MULTI-SEQUENCE SEARCH COMPLETE: Found {len(all_valid_sequences)} valid sequence(s). ---")
    
    # 1. Process sequences into the structure the UI expects: {video_id, shots}
    processed_sequences = []
    for seq in all_valid_sequences:
        if not seq: continue
        avg_score = sum(shot['rrf_score'] for shot in seq) / len(seq)
        video_id = extract_video_id(seq[0]['filepath'])
        processed_sequences.append({
            "video_id": video_id,
            "average_score": avg_score,
            "shots": seq  # The UI is looking for this 'shots' key!
        })

    # 2. Sort by the new average score
    processed_sequences.sort(key=lambda x: x['average_score'], reverse=True)

    # 3. Limit the results and add URLs
    final_sequences_data = processed_sequences[:MAX_SEQUENCES_TO_RETURN]
    base_url = str(request.base_url)
    for seq_data in final_sequences_data:
        for item in seq_data['shots']:
            encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
            item['url'] = f"{base_url}images/{encoded_path}"

    return final_sequences_data


@app.post("/search")
async def search_unified(request: Request, query_text: str = Form(None), query_image: UploadFile = File(None)):
    # ... (this function's code remains exactly the same as before)
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
    weights = {"beit3": 30, "openclip": 0.30}
    final_results = reciprocal_rank_fusion(results_for_fusion, weights)[:TOP_K_RESULTS]
    base_url = str(request.base_url)
    for item in final_results:
        encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
        item['url'] = f"{base_url}images/{encoded_path}"
    return final_results


@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    # ... (this function's code remains exactly the same as before)
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Invalid base64 encoding.")
    path_in_container = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo"
    remapped_path = original_path.replace("/workspace/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo", path_in_container, 1)
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(remapped_path)
    if not safe_path.startswith(safe_base): raise HTTPException(status_code=403, detail=f"Forbidden path. Safe base: {safe_base}, Requested path: {safe_path}")
    if not os.path.isfile(safe_path): raise HTTPException(status_code=404, detail=f"Image file not found at calculated path: {safe_path}")
    return FileResponse(safe_path)

class OcrSearchRequest(BaseModel):
    query: str

@app.post("/ocr_search")
async def ocr_search(request_data: OcrSearchRequest, request: Request):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="Chưa cung cấp từ khóa cho OCR search.")

    print("\n" + "="*50)
    print("--- BẮT ĐẦU TÌM KIẾM OCR ---")
    print(f"--- Đã nhận từ khóa OCR: '{request_data.query}' ---")

    # Gọi hàm helper đã tạo ở Bước 1.2
    results = search_ocr_on_elasticsearch(request_data.query, limit=TOP_K_RESULTS)

    if not results:
        print("--- TÌM KIẾM OCR HOÀN TẤT: Không tìm thấy kết quả. ---")
        return []

    # Thêm trường 'url' vào mỗi kết quả để giao diện hiển thị ảnh
    base_url = str(request.base_url)
    for item in results:
        encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
        item['url'] = f"{base_url}images/{encoded_path}"

    print(f"--- TÌM KIẾM OCR HOÀN TẤT: Tìm thấy {len(results)} kết quả. ---")
    return results

# NEW: ASR Search endpoint
class AsrSearchRequest(BaseModel):
    query: str

@app.post("/asr_search")
async def asr_search(request_data: AsrSearchRequest, request: Request):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="No keyword provided for ASR search.")

    print("\n" + "="*50)
    print("--- STARTING ASR SEARCH ---")
    print(f"--- Received ASR keyword: '{request_data.query}' ---")

    # Call the ASR-specific search function
    results = search_asr_on_elasticsearch(request_data.query, limit=TOP_K_RESULTS)

    if not results:
        print("--- ASR SEARCH COMPLETE: No results found. ---")
        return []

    # Add 'url' field to each result for image display
    base_url = str(request.base_url)
    for item in results:
        encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
        item['url'] = f"{base_url}images/{encoded_path}"

    print(f"--- ASR SEARCH COMPLETE: Found {len(results)} results. ---")
    return results