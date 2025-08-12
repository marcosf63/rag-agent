"""Tests for custom exceptions."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_agent.core.exceptions import (
    RagError,
    IngestionError,
    IndexNotReadyError,
    RetrievalError,
    AnswerNotFoundError,
    LLMError,
    EmbeddingError,
)


class TestExceptions:
    """Tests for custom exception hierarchy."""
    
    def test_base_exception(self):
        """Test that RagError is the base exception."""
        error = RagError("Base error")
        assert str(error) == "Base error"
        assert isinstance(error, Exception)
    
    def test_inheritance_hierarchy(self):
        """Test that all custom exceptions inherit from RagError."""
        exceptions = [
            IngestionError,
            IndexNotReadyError,
            RetrievalError,
            AnswerNotFoundError,
            LLMError,
            EmbeddingError,
        ]
        
        for exc_class in exceptions:
            error = exc_class("Test error")
            assert isinstance(error, RagError)
            assert isinstance(error, Exception)
    
    def test_exception_messages(self):
        """Test that exceptions properly store messages."""
        message = "Test error message"
        
        exceptions = [
            RagError(message),
            IngestionError(message),
            IndexNotReadyError(message),
            RetrievalError(message),
            AnswerNotFoundError(message),
            LLMError(message),
            EmbeddingError(message),
        ]
        
        for error in exceptions:
            assert str(error) == message
    
    def test_exception_raising(self):
        """Test that exceptions can be properly raised and caught."""
        with pytest.raises(IngestionError) as exc_info:
            raise IngestionError("Ingestion failed")
        assert "Ingestion failed" in str(exc_info.value)
        
        with pytest.raises(RagError):  # Should catch IngestionError too
            raise IngestionError("Ingestion failed")
    
    def test_specific_exception_types(self):
        """Test specific exception types for different error scenarios."""
        # IngestionError for document processing issues
        with pytest.raises(IngestionError):
            raise IngestionError("Failed to read PDF")
        
        # RetrievalError for vector search issues
        with pytest.raises(RetrievalError):
            raise RetrievalError("Vector search failed")
        
        # AnswerNotFoundError for cases where no relevant info is found
        with pytest.raises(AnswerNotFoundError):
            raise AnswerNotFoundError("Não encontrado nos documentos")
        
        # LLMError for language model issues
        with pytest.raises(LLMError):
            raise LLMError("OpenAI API error")
        
        # EmbeddingError for embedding generation issues
        with pytest.raises(EmbeddingError):
            raise EmbeddingError("Embedding model failed")