# utils_query.py

# =========================
# Imports
# =========================
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from typing import List
from api_key import api_key
import asyncio
import langid

# --- Import thư viện googletrans ---
# Bạn cần cài đặt thư viện này: pip install googletrans==4.0.0-rc1
from googletrans import Translator

# =========================
# Config
# =========================
MODEL_ENHANCE = "gemini-2.5-flash-lite"
MODEL_EXPAND = "gemini-2.5-flash"
NUM_EXPAND_WORKERS = 10
MAX_ATTEMPT = len(api_key)

# =========================
# API Key Rotation
# =========================
def get_client_for_thread(thread_id: int):
    """
    Lấy client theo thread_id để phân phối API key đều.
    """
    key_idx = thread_id % len(api_key)
    return genai.Client(api_key=api_key[key_idx])

# =========================
# Query Enhancement (Hàm đồng bộ)
# =========================
def enhance_query(original_query: str) -> str:
    """
    Improve a search query for better retrieval accuracy.
    Hàm này nhận đầu vào là query đã được dịch sang tiếng Anh.
    """
    for attempt in range(MAX_ATTEMPT):
        try:
            client = genai.Client(api_key=api_key[attempt % len(api_key)])
            
            prompt = (
                f"You are an expert in search query optimization for accurate and relevant retrieval.\n"
                f"Here is the original search query:\n"
                f"\"{original_query}\"\n\n"
                "Your task: Rewrite this query to maximize the chances of retrieving highly relevant results. "
                "Make the wording clear, precise, and rich in meaningful keywords. "
                "Preserve the original intent and all essential details, but remove any ambiguity or unnecessary words. "
                "If something is vague, make it more specific without changing the meaning. "
                "The query is already in English. Return only the improved English query, without any explanations."
            )

            resp = client.models.generate_content(
                model=MODEL_ENHANCE,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0
                ),
            )
            return resp.text.strip()
        except Exception:
            continue
    # Nếu enhance thất bại, trả về query gốc
    return original_query

# =========================
# Query Translate Logic (Hàm bất đồng bộ)
# =========================
# async def translate_query(query: str, dest: str = 'en') -> str:
#     """
#     Dịch truy vấn bằng cách chạy trên event loop có sẵn (của FastAPI).
#     Đây là một hàm bất đồng bộ, cần được gọi bằng 'await'.
#     """
#     if not query or not query.strip():
#         return ""

#     try:
#         # Không cần asyncio.run() nữa, vì chúng ta đang ở trong một hàm async
#         translator = Translator()
#         result = await translator.translate(query, dest=dest)
#         return result.text
#     except Exception as e:
#         print(f"--- WARNING: Google Translate failed for query '{query}'. Error: {e}. Returning original query. ---")
#         return query
async def translate_query(query: str, dest: str = 'en') -> str:
    """
    Dịch truy vấn sang 'dest' nếu không phải tiếng Anh.
    Nếu đã là tiếng Anh thì trả nguyên văn.
    """
    if not query or not query.strip():
        return ""

    try:
        # Phát hiện ngôn ngữ
        lang, _ = langid.classify(query)
        
        # Nếu đã là tiếng Anh, trả nguyên
        if lang == 'en':
            return query
        
        # Dịch nếu không phải tiếng Anh
        translator = Translator()
        result = await translator.translate(query, dest=dest)
        return result.text
    
    except Exception as e:
        print(f"--- WARNING: Google Translate failed for query '{query}'. Error: {e}. Returning original query. ---")
        return query

# =========================
# Query Expansion (Hàm đồng bộ)
# =========================
def _expand_once(short_query: str, thread_id: int) -> List[str]:
    """
    Expand a single short query into multiple scenarios.
    """
    client = get_client_for_thread(thread_id)
    prompt_text = f"""Expand the short user query into several distinct, detailed video scene descriptions. 
Each description should represent a plausible, specific scenario. 
Start each scenario on a new line with a hyphen (-).

Query: "{short_query}"
Scenarios:"""

    resp = client.models.generate_content(
        model=MODEL_EXPAND,  
        contents=prompt_text,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return [
        line.strip().lstrip('-').strip()
        for line in resp.text.strip().split('\n')
        if line.strip()
    ]

def expand_query_parallel(short_query: str, num_requests: int = NUM_EXPAND_WORKERS) -> str:
    """
    Expand query in parallel by running multiple requests concurrently.
    Output: single string with each expanded query on a new line.
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [
            executor.submit(_expand_once, short_query, i)
            for i in range(num_requests)
        ]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                pass
    # Remove duplicates & keep order
    seen = set()
    final_results = []
    for s in results:
        if s not in seen:
            seen.add(s)
            final_results.append(s)
    return "\n".join(final_results)

# =========================
# Main test (Sửa lại để chạy được hàm async)
# =========================
if __name__ == "__main__":
    # Để test một hàm async, chúng ta cần một hàm async bao bọc
    async def main_test():
        q_vi = "Một con chó đang nhai cỏ"
        q_en = "Two men in black suits"
        
        print("--- Testing Translation (async fixed) ---")
        # Dùng await để gọi hàm async
        translated_vi = await translate_query(q_vi)
        print(f"VI Input: '{q_vi}' -> Translated: '{translated_vi}'")
        
        translated_en = await translate_query(q_en)
        print(f"EN Input: '{q_en}' -> Translated: '{translated_en}'")

        print("\n--- Testing Enhancement (sync function) ---")
        enhanced_text = enhance_query("A dog is chewing grass")
        print(f"Enhancing 'A dog is chewing grass' -> '{enhanced_text}'")

    # Dùng asyncio.run() ở đây để khởi chạy toàn bộ test
    asyncio.run(main_test())