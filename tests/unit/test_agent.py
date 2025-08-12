"""Tests for the main RagAgent class."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_agent.core.agent import RagAgent
from rag_agent.core.exceptions import AnswerNotFoundError, LLMError, RetrievalError


class TestRagAgent:
    """Tests for the RagAgent class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_store = Mock()
        self.mock_llm = Mock()
        self.agent = RagAgent(
            store=self.mock_store, llm=self.mock_llm, top_k=3, distance_threshold=0.3
        )

    def test_initialization(self):
        """Test agent initialization with parameters."""
        assert self.agent.store is self.mock_store
        assert self.agent.llm is self.mock_llm
        assert self.agent.top_k == 3
        assert self.agent.distance_threshold == 0.3
        assert self.agent.max_context_chars == 4000  # default

    def test_format_prompt(self):
        """Test prompt formatting with contexts."""
        contexts = [
            ("First chunk content", {"chunk_id": 0, "source": "test.txt"}, 0.1),
            ("Second chunk content", {"chunk_id": 1, "source": "test.txt"}, 0.2),
        ]

        prompt = self.agent._format_prompt("Test question", contexts)

        assert "Test question" in prompt
        assert "First chunk content" in prompt
        assert "Second chunk content" in prompt
        assert "chunk_id=0" in prompt
        assert "chunk_id=1" in prompt
        assert "source=test.txt" in prompt
        assert "CONTEXTO INÍCIO" in prompt
        assert "CONTEXTO FIM" in prompt

    def test_successful_ask(self):
        """Test successful question answering."""
        # Mock store query response
        self.mock_store.query.return_value = (
            ["Document content 1", "Document content 2"],
            [{"chunk_id": 0, "source": "test.txt"}, {"chunk_id": 1, "source": "test.txt"}],
            [0.1, 0.2],  # Below threshold
        )

        # Mock LLM response
        self.mock_llm.answer.return_value = "This is the answer based on chunk 0 and 1"

        result = self.agent.ask("What is the content about?")

        assert result["answer"] == "This is the answer based on chunk 0 and 1"
        assert len(result["used_chunks"]) == 2
        assert result["used_chunks"][0]["chunk_id"] == 0
        assert result["used_chunks"][1]["chunk_id"] == 1
        assert "request_id" in result
        assert "latency_ms" in result

        # Verify store was called correctly
        self.mock_store.query.assert_called_once_with("What is the content about?", k=3)

    def test_no_relevant_context(self):
        """Test behavior when no relevant context is found."""
        # Mock store query with distances above threshold
        self.mock_store.query.return_value = (
            ["Some content"],
            [{"chunk_id": 0, "source": "test.txt"}],
            [0.8],  # Above threshold
        )

        with pytest.raises(AnswerNotFoundError):
            self.agent.ask("What is this about?")

    def test_llm_returns_not_found(self):
        """Test behavior when LLM returns 'not found' response."""
        self.mock_store.query.return_value = (
            ["Document content"],
            [{"chunk_id": 0, "source": "test.txt"}],
            [0.1],  # Below threshold
        )

        self.mock_llm.answer.return_value = "Não encontrado nos documentos."

        with pytest.raises(AnswerNotFoundError):
            self.agent.ask("What is this about?")

    def test_llm_returns_empty_response(self):
        """Test behavior when LLM returns empty response."""
        self.mock_store.query.return_value = (
            ["Document content"],
            [{"chunk_id": 0, "source": "test.txt"}],
            [0.1],  # Below threshold
        )

        self.mock_llm.answer.return_value = ""

        with pytest.raises(AnswerNotFoundError):
            self.agent.ask("What is this about?")

    def test_retrieval_error(self):
        """Test handling of retrieval errors."""
        self.mock_store.query.side_effect = Exception("Database error")

        with pytest.raises(RetrievalError) as exc_info:
            self.agent.ask("What is this about?")

        assert "Database error" in str(exc_info.value)

    def test_llm_error(self):
        """Test handling of LLM errors."""
        self.mock_store.query.return_value = (
            ["Document content"],
            [{"chunk_id": 0, "source": "test.txt"}],
            [0.1],
        )

        self.mock_llm.answer.side_effect = Exception("API error")

        with pytest.raises(LLMError) as exc_info:
            self.agent.ask("What is this about?")

        assert "API error" in str(exc_info.value)

    def test_custom_request_id(self):
        """Test that custom request ID is used."""
        self.mock_store.query.return_value = (
            ["Document content"],
            [{"chunk_id": 0, "source": "test.txt"}],
            [0.1],
        )

        self.mock_llm.answer.return_value = "Answer"

        result = self.agent.ask("Question", request_id="custom-123")

        assert result["request_id"] == "custom-123"

    def test_context_size_limiting(self):
        """Test that context is limited by max_context_chars."""
        # Create agent with small context limit
        small_agent = RagAgent(
            store=self.mock_store,
            llm=self.mock_llm,
            max_context_chars=150,  # Small but reasonable limit
        )

        # Create large contexts
        large_content = "A" * 80  # Reduced size
        contexts = [
            (large_content, {"chunk_id": 0, "source": "test.txt"}, 0.1),
            (large_content, {"chunk_id": 1, "source": "test.txt"}, 0.2),
        ]

        prompt = small_agent._format_prompt("Question", contexts)

        # Check that only first context is included due to size limit
        # The prompt should contain the first context but not both
        assert "chunk_id=0" in prompt
        # Count of "A" characters should be less than 160 (2 * 80)
        assert prompt.count("A") <= 80
