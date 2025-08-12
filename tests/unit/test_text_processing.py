"""Tests for text processing utilities."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_agent.utils.text_processing import chunk_text
from rag_agent.core.exceptions import IngestionError


class TestChunkText:
    """Tests for the chunk_text function."""
    
    def test_basic_chunking(self):
        """Test basic text chunking functionality."""
        text = "This is a simple test text that should be chunked properly."
        chunks = chunk_text(text, max_chars=20, overlap=5)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)
    
    def test_short_text_single_chunk(self):
        """Test that short text produces a single chunk."""
        text = "Short text"
        chunks = chunk_text(text, max_chars=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_exact_size_text(self):
        """Test text that exactly matches max_chars."""
        text = "A" * 50
        chunks = chunk_text(text, max_chars=50, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_overlap_functionality(self):
        """Test that overlap works correctly."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, max_chars=10, overlap=3)
        
        assert len(chunks) > 1
        # Check that chunks have expected overlap
        for i in range(len(chunks) - 1):
            current_end = chunks[i][-3:]  # last 3 chars
            next_start = chunks[i + 1][:3]  # first 3 chars
            assert current_end == next_start
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise IngestionError."""
        text = "Test text"
        
        # max_chars <= overlap should raise error
        with pytest.raises(IngestionError):
            chunk_text(text, max_chars=10, overlap=10)
        
        with pytest.raises(IngestionError):
            chunk_text(text, max_chars=5, overlap=10)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        text = ""
        chunks = chunk_text(text, max_chars=100, overlap=10)
        
        # Empty text should return empty list, not single empty chunk
        assert len(chunks) == 0
    
    def test_whitespace_preservation(self):
        """Test that whitespace is preserved in chunks."""
        text = "Word1 Word2  Word3\nWord4\tWord5"
        chunks = chunk_text(text, max_chars=15, overlap=5)
        
        # Reconstruct text from chunks (accounting for overlap)
        full_text = chunks[0]
        for chunk in chunks[1:]:
            full_text += chunk[5:]  # Skip overlap
        
        assert full_text == text