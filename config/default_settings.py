"""Default configuration settings for RAG Agent."""

from typing import Dict, Any


DEFAULT_EMBEDDING_CONFIG = {
    "openai": {
        "model": "text-embedding-3-small",
    },
    "sentence_transformers": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
}

DEFAULT_LLM_CONFIG = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
    "ollama": {
        "model": "llama3.1:8b",
        "host": "http://localhost:11434",
        "temperature": 0.0,
        "timeout": 120,
    },
}

DEFAULT_VECTOR_STORE_CONFIG = {
    "chroma": {
        "persist_dir": "./chroma_db",
        "collection_name": "rag_documents",
        "distance_metric": "cosine",
    },
}

DEFAULT_AGENT_CONFIG = {
    "top_k": 5,
    "max_context_chars": 4000,
    "distance_threshold": 0.35,
}

DEFAULT_INGESTION_CONFIG = {
    "chunk_size": 1200,
    "chunk_overlap": 120,
    "supported_formats": [".txt", ".md", ".pdf"],
}

DEFAULT_LOGGING_CONFIG = {
    "level": "INFO",
    "format": "json",
    "logger_name": "rag",
}


def get_default_config() -> Dict[str, Any]:
    """Get the complete default configuration."""
    return {
        "embedding": DEFAULT_EMBEDDING_CONFIG,
        "llm": DEFAULT_LLM_CONFIG,
        "vector_store": DEFAULT_VECTOR_STORE_CONFIG,
        "agent": DEFAULT_AGENT_CONFIG,
        "ingestion": DEFAULT_INGESTION_CONFIG,
        "logging": DEFAULT_LOGGING_CONFIG,
    }