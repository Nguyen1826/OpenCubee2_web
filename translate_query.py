# File: translate_query.py
import sys
import os
import langdetect
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
import time
from google import genai
from api_key import api_key

def get_expanded_queries(short_query):
    client = get_client_cycle_api()
    prompt_text = f"""Expand the short user query into several distinct, detailed video scene descriptions. Each description should represent a plausible, specific scenario. Start each scenario on a new line with a hyphen (-).

                Query: "Making coffee"
                Scenarios:
                - A close-up shot of a barista creating latte art on an espresso.
                - A time-lapse of a cold brew coffee maker dripping.
                - A tutorial showing someone using a French press at home.

                Query: "Dog contest"
                Scenarios:
                - A handler guiding a Poodle around a show ring at the Westminster Dog Show.
                - A Border Collie racing through a tunnel in an agility competition.
                - A funny clip of a Corgi in a costume contest.

                Query: "{short_query}"
                Scenarios:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",  
        contents=prompt_text,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    scenarios = [line.strip().lstrip('-').strip() for line in response.text.strip().split('\n') if line.strip()]
    return scenarios
    
MAX_ATTEMPT = len(api_key)
cur = 0
def get_english_query(vietnamese_query):
    global cur
    
    for _ in range(MAX_ATTEMPT):
        try:
            client = genai.Client(api_key=api_key[cur])
            prompt = f"Translating this Vietnamese Query to English Query but Keeping the meaning as most as possible, make the query easier to understand, wrap the translated result in asterisks (*...*). Vietnamese Query: {vietnamese_query}"
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            
            return response.text.split("*")[1]

        except Exception as e:
            print(f"Error: {e}")
            print(f"Attempting {cur}th key!")
            cur = (cur + 1) % MAX_ATTEMPT

MAX_ATTEMPT = len(api_key)
cur = 0
def enhance_query(original_query):
    global cur
    
    for _ in range(MAX_ATTEMPT):
        try:
            client = genai.Client(api_key=api_key[cur])
            prompt = (
                f"You are an expert in search query optimization for accurate and relevant retrieval.\n"
                f"Here is the original search query:\n"
                f"\"{original_query}\"\n\n"
                "Your task: Rewrite this query to maximize the chances of retrieving highly relevant results. "
                "Make the wording clear, precise, and rich in meaningful keywords. "
                "Preserve the original intent and all essential details, but remove any ambiguity or unnecessary words. "
                "If something is vague, make it more specific without changing the meaning. "
                "Return only the improved query, without any explanations"
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0
                ),
            )
            
            print(response.text)
            return response.text

        except Exception as e:
            print(f"Error: {e}")
            print(f"Attempting {cur}th key!")
            cur = (cur + 1) % MAX_ATTEMPT

def translate_text(query: str) -> str:
    """
    Translates a query to English, but only if it's detected as Vietnamese.
    """
    try:
        # Guard Clause: If not Vietnamese, return immediately to save time.
        if langdetect.detect(query) != 'vi':
            return query
    except Exception:
        # If detection fails, it's safer to return the original query.
        return query
    # Call your LLM function only for Vietnamese queries.
    return get_english_query(query)

def enhancing(query: str) -> str:
    return enhance_query(query)
def expanding(query: str) -> str:
    return get_expanded_queries(query)[2:]
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_query = " ".join(sys.argv[1:])
        translated_query = translate_text(input_query)
        print(translated_query)