"""Custom exceptions for the RAG Agent."""


class RagError(Exception):
    """Base exception for all RAG-related errors."""
    pass


class IngestionError(RagError):
    """Raised when document ingestion fails."""
    pass


class IndexNotReadyError(RagError):
    """Raised when vector index is not ready for queries."""
    pass


class RetrievalError(RagError):
    """Raised when document retrieval fails."""
    pass


class AnswerNotFoundError(RagError):
    """Raised when no relevant information is found in documents."""
    pass


class LLMError(RagError):
    """Raised when language model generation fails."""
    pass


class EmbeddingError(RagError):
    """Raised when embedding generation fails."""
    pass