"""
RAG Agent - Retrieval-Augmented Generation Agent

A strict document-based question answering system using ChromaDB and pluggable providers.
"""

from .core.agent import RagAgent
from .core.exceptions import (
    AnswerNotFoundError,
    EmbeddingError,
    IndexNotReadyError,
    IngestionError,
    LLMError,
    RagError,
    RetrievalError,
)
from .providers.embeddings import OpenAIEmbedding, SentenceTransformerEmbedding
from .providers.llm import OllamaChat, OpenAIChat
from .storage.chroma_store import ChromaStore
from .utils.ingestion import ingest_file, read_text_from_path
from .utils.logging import setup_logger
from .utils.text_processing import chunk_text

__version__ = "0.1.0"

__all__ = [
    "RagAgent",
    "RagError",
    "IngestionError",
    "IndexNotReadyError",
    "RetrievalError",
    "AnswerNotFoundError",
    "LLMError",
    "EmbeddingError",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "OpenAIChat",
    "OllamaChat",
    "ChromaStore",
    "ingest_file",
    "read_text_from_path",
    "chunk_text",
    "setup_logger",
]
