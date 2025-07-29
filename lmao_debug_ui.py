# FILE: gateway_server_debug.py
# PURPOSE: A version of the gateway with ENHANCED LOGGING and a more transparent
# response structure to easily diagnose issues with the temporal search.

import os
import sys
import traceback
import base64
import asyncio
from collections import defaultdict
import httpx
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from pymilvus import Collection, connections, utility

# --- Path setup remains the same ---
COMMON_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)

# --- Import and debug block remains the same ---
try:
    from Cubi.translate_query import translate_text
    print("--- [DEBUG] Gateway Server: Successfully imported 'translate_text' from Cubi subfolder. ---")
except ImportError as e:
    print("\n" + "="*80)
    print("!!! [DEBUG] FATAL IMPORT ERROR: Could not import 'translate_text'. !!!")
    traceback.print_exc()
    print("="*80 + "\n")


# --- Configuration ---
BEIT3_WORKER_URL = "http://model-workers:8001/embed"
OPENCLIP_WORKER_URL = "http://model-workers:8002/embed"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
BEIT3_COLLECTION = "beit3_large_embeddings"
OPENCLIP_COLLECTION = "openclip_h14_embeddings"
SEARCH_DEPTH = 500
TOP_K_RESULTS = 50
ALLOWED_BASE_DIR = COMMON_PARENT_DIR
MAX_SEQUENCES_TO_RETURN = 50
INITIAL_CANDIDATES = 50
SEARCH_DEPTH_PER_STAGE = 200


# --- FastAPI App Initialization & Startup ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("--- [DEBUG] Gateway Server: Connecting to Milvus... ---")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- [DEBUG] Gateway Server: Milvus connection successful. ---")
        # Check collections
        if utility.has_collection(BEIT3_COLLECTION):
            print(f"--- [DEBUG] Milvus check: Collection '{BEIT3_COLLECTION}' found.")
        else:
            print(f"--- [DEBUG] WARNING: Collection '{BEIT3_COLLECTION}' NOT FOUND in Milvus.")
        if utility.has_collection(OPENCLIP_COLLECTION):
            print(f"--- [DEBUG] Milvus check: Collection '{OPENCLIP_COLLECTION}' found.")
        else:
            print(f"--- [DEBUG] WARNING: Collection '{OPENCLIP_COLLECTION}' NOT FOUND in Milvus.")

    except Exception as e:
        print(f"!!! [DEBUG] FATAL: Could not connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}. Error: {e}")
        traceback.print_exc()

# --- Helper Functions (with added logging) ---
def extract_video_id(filepath: str) -> str:
    match = re.search(r'.*/(L\d+_V\d+)_', filepath)
    if match:
        return match.group(1)
    return None

def search_milvus(collection_name: str, query_vector, limit: int, min_filepath: str = None, debug_log: list = None):
    log_func = lambda msg: debug_log.append(msg) if debug_log is not None else print(msg)
    
    log_func(f"    - Searching Milvus collection: '{collection_name}' with limit: {limit}")
    try:
        if not utility.has_collection(collection_name):
            log_func(f"    - ERROR: Collection '{collection_name}' does not exist.")
            return []
        collection = Collection(collection_name)
        collection.load()
        expr = f"filepath > '{min_filepath}'" if min_filepath else None
        log_func(f"    - Milvus Query Filter (expr): {expr or 'None'}")
        
        search_params = {"metric_type": "COSINE", "params": {"ef": max(256, limit + 50)}}
        results = collection.search(query_vector, "embedding", search_params, limit=limit, output_fields=["filepath"], expr=expr)
        
        if not results or not results[0]:
            log_func("    - Milvus returned no results.")
            return []
            
        hits = [{"filepath": hit.entity.get("filepath"), "score": hit.distance} for hit in results[0]]
        log_func(f"    - Milvus returned {len(hits)} hits.")
        return hits
    except Exception as e:
        log_func(f"    - !!! FATAL ERROR during Milvus search on '{collection_name}': {e}")
        traceback.print_exc()
        return []

def reciprocal_rank_fusion(results_lists: dict, weights: dict, k_rrf: int = 60):
    # This function is pure logic, less need for debug logs unless results are unexpected
    rrf_scores = defaultdict(float)
    raw_scores = defaultdict(dict)
    for model_name, results in results_lists.items():
        similarity_results = []
        for result in results:
            similarity_results.append({'filepath': result['filepath'], 'score': max(0, 1.0 - result.get('score', 1.0))})

        for rank, result in enumerate(similarity_results, 1):
            filepath = result['filepath']
            rrf_scores[filepath] += weights[model_name] * (1.0 / (k_rrf + rank))
            raw_scores[filepath][model_name] = result.get('score', 0.0)
    
    fused_results = [{"filepath": fp, "rrf_score": score, "beit3_sim": raw_scores[fp].get("beit3", 0.0), "openclip_sim": raw_scores[fp].get("openclip", 0.0)} for fp, score in rrf_scores.items()]
    return sorted(fused_results, key=lambda x: x['rrf_score'], reverse=True)


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    A "bulletproof" version of the root endpoint.
    It catches ALL exceptions during file path checking and reading,
    and reports them clearly in HTML format.
    """
    # HTML template for displaying errors
    def get_error_html(title, message, details):
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gateway Error</title>
            <style>
                body {{ font-family: sans-serif; margin: 2em; color: #333; }}
                .container {{ max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
                h1 {{ color: #d9534f; }}
                code {{ background-color: #eee; padding: 3px 5px; border-radius: 3px; border: 1px solid #ccc; }}
                pre {{ background-color: #eee; padding: 10px; border-radius: 3px; white-space: pre-wrap; word-wrap: break-word; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <p><strong>Thông báo:</strong> {message}</p>
                <hr>
                <h2>Chi tiết Gỡ lỗi:</h2>
                {details}
            </div>
        </body>
        </html>
        """

    try:
        # 1. Construct the path
        ui_path = os.path.join(ALLOWED_BASE_DIR, "Cubi", "ui", "ui1_2_temporal.html")
        cwd = os.getcwd()

        details_html = f"""
            <p>Thư mục làm việc hiện tại (CWD):</p>
            <code>{cwd}</code>
            <p>Biến ALLOWED_BASE_DIR được tính là:</p>
            <code>{ALLOWED_BASE_DIR}</code>
            <p>Đường dẫn đầy đủ đến tệp UI được ghép lại là:</p>
            <code>{ui_path}</code>
        """

        # 2. Check if path exists
        if not os.path.exists(ui_path):
            title = "Lỗi 404: Không tìm thấy đường dẫn"
            message = "Đường dẫn đến tệp giao diện người dùng không tồn tại."
            return HTMLResponse(content=get_error_html(title, message, details_html), status_code=404)

        # 3. Check if it's a file, not a directory
        if not os.path.isfile(ui_path):
            title = "Lỗi 500: Đường dẫn không phải là tệp"
            message = "Đường dẫn đã được tìm thấy, nhưng nó là một thư mục hoặc một loại khác, không phải là một tệp thông thường."
            return HTMLResponse(content=get_error_html(title, message, details_html), status_code=500)

        # 4. Try to open and read the file
        try:
            with open(ui_path, "r", encoding="utf-8") as f:
                content = f.read()
            # If successful, return the content
            return HTMLResponse(content=content)
        except Exception as e:
            # This catches file permission errors, encoding errors, etc.
            title = "Lỗi 500: Không thể đọc tệp"
            message = "Đã tìm thấy tệp, nhưng không thể đọc nội dung của nó. Vấn đề có thể do quyền truy cập (permission) hoặc lỗi mã hóa (encoding)."
            details_html += f"""
                <h3>Lỗi cụ thể từ Python:</h3>
                <pre>{traceback.format_exc()}</pre>
            """
            return HTMLResponse(content=get_error_html(title, message, details_html), status_code=500)

    except Exception as e:
        # This is the final safety net, catching any other unexpected error
        title = "Lỗi 500: Lỗi không xác định trong Endpoint"
        message = "Một lỗi không mong muốn đã xảy ra khi đang cố gắng phục vụ trang chủ."
        details_html = f"<pre>{traceback.format_exc()}</pre>"
        return HTMLResponse(content=get_error_html(title, message, details_html), status_code=500)

class TemporalSearchRequest(BaseModel):
    queries: list[str]

@app.post("/temporal_search")
async def temporal_search_debug(request_data: TemporalSearchRequest, request: Request):
    # This is the debug version of the temporal search endpoint.
    # It will return a JSON object with a detailed log of its operations.
    
    debug_log = []
    queries = request_data.queries
    
    debug_log.append("="*50)
    debug_log.append("--- [DEBUG] INITIATING MULTI-SEQUENCE TEMPORAL SEARCH ---")
    
    if not queries:
        debug_log.append("--- [DEBUG] FAILED: No queries provided in the request.")
        return JSONResponse(status_code=400, content={
            "status": "Thất bại",
            "message": "Không có truy vấn nào được cung cấp.",
            "debug_log": debug_log,
            "data": []
        })

    debug_log.append(f"--- [DEBUG] Received {len(queries)} stages. Finding up to {INITIAL_CANDIDATES} initial paths.")
    debug_log.append(f"--- [DEBUG] Queries: {queries}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # --- Stage 1: Get initial candidates ---
        debug_log.append(f"\n--- [DEBUG] Processing Stage 1: '{queries[0]}' ---")
        
        translated_query = translate_text(queries[0])
        debug_log.append(f"--- [DEBUG] Translated query: '{translated_query}'")
        
        debug_log.append("--- [DEBUG] Sending requests to BEiT-3 and OpenCLIP workers...")
        vec_responses = await asyncio.gather(
            client.post(BEIT3_WORKER_URL, data={'text_query': translated_query}),
            client.post(OPENCLIP_WORKER_URL, data={'text_query': translated_query}),
            return_exceptions=True
        )

        # Robust checking of worker responses
        if isinstance(vec_responses[0], Exception) or vec_responses[0].status_code != 200:
            error_detail = str(vec_responses[0]) if isinstance(vec_responses[0], Exception) else f"Status {vec_responses[0].status_code}: {vec_responses[0].text}"
            debug_log.append(f"--- [DEBUG] !!! FAILED: BEiT-3 worker returned an error: {error_detail}")
            return JSONResponse(status_code=500, content={"status": "Thất bại", "message": "BEiT-3 worker gặp lỗi.", "debug_log": debug_log, "data": []})

        if isinstance(vec_responses[1], Exception) or vec_responses[1].status_code != 200:
            error_detail = str(vec_responses[1]) if isinstance(vec_responses[1], Exception) else f"Status {vec_responses[1].status_code}: {vec_responses[1].text}"
            debug_log.append(f"--- [DEBUG] !!! FAILED: OpenCLIP worker returned an error: {error_detail}")
            return JSONResponse(status_code=500, content={"status": "Thất bại", "message": "OpenCLIP worker gặp lỗi.", "debug_log": debug_log, "data": []})
        
        debug_log.append("--- [DEBUG] Both workers returned success. Extracting embeddings.")
        vec_beit3 = vec_responses[0].json()['embedding']
        vec_opc = vec_responses[1].json()['embedding']

        beit3_results = search_milvus(BEIT3_COLLECTION, [vec_beit3], SEARCH_DEPTH_PER_STAGE, debug_log=debug_log)
        openclip_results = search_milvus(OPENCLIP_COLLECTION, [vec_opc], SEARCH_DEPTH_PER_STAGE, debug_log=debug_log)

        initial_candidates = reciprocal_rank_fusion({"beit3": beit3_results, "openclip": openclip_results}, {"beit3": 30, "openclip": 0.30})[:INITIAL_CANDIDATES]

        if not initial_candidates:
            debug_log.append("--- [DEBUG] !!! SEARCH FAILED: No candidates found for the first stage after fusion.")
            return JSONResponse(status_code=200, content={"status": "Không tìm thấy", "message": "Không tìm thấy ứng viên nào cho giai đoạn đầu tiên.", "debug_log": debug_log, "data": []})
        
        debug_log.append(f"--- [DEBUG] Found {len(initial_candidates)} potential starting shots. Starting to branch out.")

        # --- Subsequent Stages ---
        all_valid_sequences = []
        for i, candidate in enumerate(initial_candidates):
            video_id = extract_video_id(candidate['filepath'])
            if not video_id:
                debug_log.append(f"  - WARNING: Could not extract video_id from candidate filepath: {candidate['filepath']}")
                continue

            debug_log.append(f"\n  -> [Path {i+1}/{len(initial_candidates)}] Branching from candidate: '{os.path.basename(candidate['filepath'])}' (Video: {video_id})")
            
            current_sequence = [candidate]
            last_successful_path = candidate['filepath']
            is_complete = True

            # If there's only one query, the path is complete by default
            if len(queries) == 1:
                debug_log.append(f"  -> SUCCESS: Only one stage required. Sequence found.")
                all_valid_sequences.append(current_sequence)
                continue

            for stage_idx, query in enumerate(queries[1:]):
                stage_num = stage_idx + 2
                debug_log.append(f"    -- [Path {i+1}] Searching for Stage {stage_num}: '{query}'")
                
                t_query = translate_text(query)
                # No need to log worker calls again, they are less likely to fail here if they worked before
                vecs = await asyncio.gather(
                    client.post(BEIT3_WORKER_URL, data={'text_query': t_query}),
                    client.post(OPENCLIP_WORKER_URL, data={'text_query': t_query}),
                    return_exceptions=True # Keep this for safety
                )
                
                # Check for worker errors on this path
                if isinstance(vecs[0], Exception) or isinstance(vecs[1], Exception):
                    debug_log.append(f"    -- !!! Path {i+1} broken. A model worker failed for Stage {stage_num}. Skipping this path.")
                    is_complete = False
                    break
                
                b_vec, o_vec = vecs[0].json()['embedding'], vecs[1].json()['embedding']
                
                b_res = search_milvus(BEIT3_COLLECTION, [b_vec], SEARCH_DEPTH_PER_STAGE, min_filepath=last_successful_path, debug_log=debug_log)
                o_res = search_milvus(OPENCLIP_COLLECTION, [o_vec], SEARCH_DEPTH_PER_STAGE, min_filepath=last_successful_path, debug_log=debug_log)
                
                fused_results = reciprocal_rank_fusion({"beit3": b_res, "openclip": o_res}, {"beit3": 30, "openclip": 0.30})
                debug_log.append(f"    -- [Path {i+1}] Fused {len(fused_results)} results for Stage {stage_num}.")

                # This is the most critical filtering step
                filtered_by_video = [res for res in fused_results if extract_video_id(res['filepath']) == video_id]
                debug_log.append(f"    -- [Path {i+1}] After filtering for video '{video_id}', {len(filtered_by_video)} results remain.")

                if not filtered_by_video:
                    debug_log.append(f"    -- !!! Path {i+1} broken. No matching result found for Stage {stage_num} in Video {video_id}.")
                    is_complete = False
                    break
                
                top_result = filtered_by_video[0]
                current_sequence.append(top_result)
                last_successful_path = top_result['filepath']
                debug_log.append(f"    -- [Path {i+1}] Found Stage {stage_num} match: '{os.path.basename(last_successful_path)}'")

            if is_complete:
                debug_log.append(f"  -> SUCCESS: Found a complete sequence of {len(current_sequence)} shots for Path {i+1}.")
                all_valid_sequences.append(current_sequence)
    
    # --- Finalization ---
    debug_log.append(f"\n--- [DEBUG] MULTI-SEQUENCE SEARCH COMPLETE: Found {len(all_valid_sequences)} valid sequence(s) in total. ---")
    
    processed_sequences = []
    for seq in all_valid_sequences:
        if not seq: continue
        avg_score = sum(shot['rrf_score'] for shot in seq) / len(seq)
        video_id = extract_video_id(seq[0]['filepath'])
        processed_sequences.append({
            "video_id": video_id,
            "average_score": avg_score,
            "shots": seq
        })

    processed_sequences.sort(key=lambda x: x['average_score'], reverse=True)
    final_sequences_data = processed_sequences[:MAX_SEQUENCES_TO_RETURN]
    
    base_url = str(request.base_url)
    for seq_data in final_sequences_data:
        for item in seq_data['shots']:
            encoded_path = base64.urlsafe_b64encode(item['filepath'].encode('utf-8')).decode('utf-8')
            item['url'] = f"{base_url}images/{encoded_path}"

    message = f"Tìm thấy {len(final_sequences_data)} chuỗi kết quả hợp lệ." if final_sequences_data else "Không tìm thấy chuỗi kết quả nào đáp ứng đủ tất cả các giai đoạn."
    
    return JSONResponse(status_code=200, content={
        "status": "Thành công" if final_sequences_data else "Không tìm thấy",
        "message": message,
        "debug_log": debug_log,
        "data": final_sequences_data
    })


# The /search and /images endpoints remain the same, as they are less likely to be the problem source.
@app.post("/search")
async def search_unified(request: Request, query_text: str = Form(None), query_image: UploadFile = File(None)):
    if not query_text and not query_image: raise HTTPException(status_code=400, detail="Provide query_text or query_image")
    final_query_text_for_models = query_text
    if query_text:
        translated_query = translate_text(query_text)
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
    beit3_results = search_milvus(BEIT3_COLLECTION, [vec_beit3], SEARCH_DEPTH)
    openclip_results = search_milvus(OPENCLIP_COLLECTION, [vec_opc], SEARCH_DEPTH)
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
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Invalid base64 encoding.")
    path_in_container = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo"
    remapped_path = original_path.replace("/workspace/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo", path_in_container, 1)
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(remapped_path)
    if not safe_path.startswith(safe_base): raise HTTPException(status_code=403, detail=f"Forbidden path. Safe base: {safe_base}, Requested path: {safe_path}")
    if not os.path.isfile(safe_path): raise HTTPException(status_code=404, detail=f"Image file not found at calculated path: {safe_path}")
    return FileResponse(safe_path)