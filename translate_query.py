# File: translate_query.py
import sys
import os
import langdetect

# Add the project root to the system path to allow for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import your custom LLM function
from Enn.llm.main import get_english_query, enhance_query, get_expanded_queries

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