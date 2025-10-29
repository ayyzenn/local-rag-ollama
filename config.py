"""Configuration settings for Agentic RAG System"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Using gemini-2.5-flash (latest stable) or gemini-flash-latest as fallback
# Model names should match what's available in the API (with or without models/ prefix)
GEMINI_MODEL = "gemini-2.5-flash"

# Agent Parameters
AGENT_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
}

# Router Agent Settings
ROUTER_CONFIG = {
    "temperature": 0.3,  # Lower for more deterministic routing
    "max_output_tokens": 256,
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    "collection_name": "knowledge_base",
    "embedding_model": "all-MiniLM-L6-v2",
}

# Basic Generator Settings
BASIC_GENERATOR_CONFIG = {
    "n_results": 3,  # Number of chunks to retrieve
}

# Advanced Generator Settings
ADVANCED_GENERATOR_CONFIG = {
    "query_decomposition": {
        "max_sub_queries": 5,
        "n_results_per_query": 2,
    },
    "hyde": {
        "n_results": 5,
    },
    "multi_query": {
        "n_variations": 4,
        "n_results_per_variation": 2,
    },
}

# Chunking Settings
CHUNK_CONFIG = {
    "max_words": 100,
    "overlap_words": 20,  # 20% overlap
}

# Answer Evaluation Thresholds
EVALUATION_CONFIG = {
    "min_confidence_score": 0.6,
    "min_completeness_score": 0.7,
}

