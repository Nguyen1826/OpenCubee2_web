# utils_query.py
import langdetect
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from typing import List
from api_key import api_key

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
# Query Enhancement (Có thể kèm dịch sang tiếng Anh)
# =========================
def enhance_query(original_query: str, force_translate_to_en: bool = False) -> str:
    """
    Improve a search query for better retrieval accuracy.
    Nếu force_translate_to_en=True thì yêu cầu model dịch sang tiếng Anh trước khi enhance.
    """
    for attempt in range(MAX_ATTEMPT):
        try:
            client = genai.Client(api_key=api_key[attempt % len(api_key)])

            if force_translate_to_en:
                prompt = (
                    f"You are an expert in search query optimization for accurate and relevant retrieval.\n"
                    f"The original search query is in a non-English language:\n"
                    f"\"{original_query}\"\n\n"
                    "First, translate this query into clear and natural English while keeping the meaning exactly the same. "
                    "Then, rewrite the translated query to maximize the chances of retrieving highly relevant results. "
                    "Make the wording clear, precise, and rich in meaningful keywords. "
                    "Preserve the original intent and all essential details, but remove any ambiguity or unnecessary words. "
                    "If something is vague, make it more specific without changing the meaning. "
                    "Return only the improved English query, without any explanations."
                )
            else:
                prompt = (
                    f"You are an expert in search query optimization for accurate and relevant retrieval.\n"
                    f"Here is the original search query:\n"
                    f"\"{original_query}\"\n\n"
                    "Your task: Rewrite this query to maximize the chances of retrieving highly relevant results. "
                    "Make the wording clear, precise, and rich in meaningful keywords. "
                    "Preserve the original intent and all essential details, but remove any ambiguity or unnecessary words. "
                    "If something is vague, make it more specific without changing the meaning. "
                    "Return only the improved query, without any explanations."
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
    return original_query

# =========================
# Query Translate + Enhance Logic
# =========================
def translate_query(query: str) -> str:
    """
    Nếu là tiếng Anh → trả nguyên văn.
    Nếu không phải tiếng Anh → enhance kèm yêu cầu dịch sang tiếng Anh trong prompt.
    """
    try:
        lang = langdetect.detect(query)
    except Exception:
        lang = "unknown"

    if lang != "vi":
        return query  # đã là tiếng Anh
    else:
        return enhance_query(query, force_translate_to_en=True)

# =========================
# Query Expansion
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
# Main test
# =========================
if __name__ == "__main__":
    q_vi = "Em bé đang khóc"
    q_en = "Two men in black suits appear in a newspaper, one is holding a microphone"

    print("VI input ->", translate_query(q_vi))
    print("EN input ->", translate_query(q_en))
    print("Expand Parallel:\n", expand_query_parallel("Making coffee"))
