import os

# =====================================================================
# CONFIGURATION FILE FOR GEMINI API KEY & RAG PIPELINE SETTINGS
# =====================================================================

# Paste your Gemini API Key directly below inside the quotes:
GEMINI_API_KEY = "AQ.Ab8RN6LlRZK5i9APM_vcgCW53Xx9DvbAbEbeWzrWvmgLY45r8A"

# Model choice:
MODEL_NAME = "gemini-3.6-flash"

# RAG Settings:
TOP_K_EXAMPLES = 3       # Number of retrieved dynamic few-shot training examples
BM25_WEIGHT = 0.6        # Weight for BM25 lexical search score
TFIDF_WEIGHT = 0.4       # Weight for TF-IDF similarity score

def get_api_key() -> str:
    """
    Returns API key from config.py variable or environment variable.
    """
    key = GEMINI_API_KEY.strip()
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    return key
