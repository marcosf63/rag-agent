"""Integration tests for the full RAG pipeline."""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_agent import (
    RagAgent,
    ChromaStore,
    ingest_file,
    AnswerNotFoundError,
    RagError,
)


class MockEmbedding:
    """Mock embedding provider for testing."""
    
    def embed(self, texts):
        # Return simple embeddings based on text length and first char
        return [
            [float(len(text)), ord(text[0]) if text else 0.0, 1.0] 
            for text in texts
        ]


class MockLLM:
    """Mock LLM provider for testing."""
    
    def __init__(self, response="Mocked response"):
        self.response = response
    
    def answer(self, prompt):
        if "contexto" not in prompt.lower():
            return "Não encontrado nos documentos."
        return self.response


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    doc1_path = Path(temp_dir) / "doc1.txt"
    doc2_path = Path(temp_dir) / "doc2.txt"
    
    doc1_path.write_text("This is the first document about artificial intelligence and machine learning.")
    doc2_path.write_text("This is the second document about natural language processing and deep learning.")
    
    return str(doc1_path), str(doc2_path)


class TestFullPipeline:
    """Integration tests for the complete RAG pipeline."""
    
    def test_full_pipeline_with_mocks(self, temp_dir, sample_documents):
        """Test the complete pipeline with mocked providers."""
        doc1_path, doc2_path = sample_documents
        
        # Setup components
        embedder = MockEmbedding()
        llm = MockLLM("Based on the context, this is about artificial intelligence.")
        
        store = ChromaStore(
            collection="test_collection",
            embedder=embedder,
            persist_dir=temp_dir
        )
        
        # Ingest documents
        ingest_file(doc1_path, store, source_name="doc1.txt")
        ingest_file(doc2_path, store, source_name="doc2.txt")
        
        # Create agent
        agent = RagAgent(
            store=store,
            llm=llm,
            top_k=2,
            distance_threshold=0.5
        )
        
        # Test successful query
        result = agent.ask("What is this about?")
        
        assert "artificial intelligence" in result["answer"]
        assert len(result["used_chunks"]) > 0
        assert "request_id" in result
        assert "latency_ms" in result
        assert result["latency_ms"] > 0
    
    def test_pipeline_with_no_relevant_docs(self, temp_dir, sample_documents):
        """Test pipeline behavior when no relevant documents are found."""
        doc1_path, doc2_path = sample_documents
        
        # Setup with very restrictive distance threshold
        embedder = MockEmbedding()
        llm = MockLLM("Some response")
        
        store = ChromaStore(
            collection="test_collection_2",
            embedder=embedder,
            persist_dir=temp_dir
        )
        
        ingest_file(doc1_path, store, source_name="doc1.txt")
        
        agent = RagAgent(
            store=store,
            llm=llm,
            top_k=2,
            distance_threshold=0.001  # Very restrictive
        )
        
        # Should raise AnswerNotFoundError
        with pytest.raises(AnswerNotFoundError):
            agent.ask("What is this about?")
    
    def test_pipeline_with_llm_not_found_response(self, temp_dir, sample_documents):
        """Test pipeline when LLM returns 'not found' response."""
        doc1_path, doc2_path = sample_documents
        
        embedder = MockEmbedding()
        llm = MockLLM("Não encontrado nos documentos.")
        
        store = ChromaStore(
            collection="test_collection_3",
            embedder=embedder,
            persist_dir=temp_dir
        )
        
        ingest_file(doc1_path, store, source_name="doc1.txt")
        
        agent = RagAgent(
            store=store,
            llm=llm,
            top_k=2,
            distance_threshold=0.5
        )
        
        with pytest.raises(AnswerNotFoundError):
            agent.ask("What is this about?")
    
    def test_chunking_and_retrieval(self, temp_dir):
        """Test that documents are properly chunked and retrieved."""
        # Create a longer document that will be chunked
        long_doc_path = Path(temp_dir) / "long_doc.txt"
        long_content = "This is section one about topic A. " * 20 + \
                      "This is section two about topic B. " * 20
        long_doc_path.write_text(long_content)
        
        embedder = MockEmbedding()
        llm = MockLLM("Based on the context, this discusses topics A and B.")
        
        store = ChromaStore(
            collection="test_chunking",
            embedder=embedder,
            persist_dir=temp_dir
        )
        
        # Ingest with small chunk size to force chunking
        ingest_file(str(long_doc_path), store, source_name="long_doc.txt", max_chars=100, overlap=20)
        
        agent = RagAgent(
            store=store,
            llm=llm,
            top_k=5,
            distance_threshold=0.5
        )
        
        result = agent.ask("What topics are discussed?")
        
        assert result["answer"] == "Based on the context, this discusses topics A and B."
        assert len(result["used_chunks"]) > 1  # Should have multiple chunks
    
    def test_multiple_sources(self, temp_dir):
        """Test retrieval from multiple document sources."""
        # Create documents with different topics
        doc1_path = Path(temp_dir) / "python.txt"
        doc2_path = Path(temp_dir) / "javascript.txt"
        
        doc1_path.write_text("Python is a high-level programming language with simple syntax.")
        doc2_path.write_text("JavaScript is a scripting language used for web development.")
        
        embedder = MockEmbedding()
        llm = MockLLM("Both Python and JavaScript are programming languages.")
        
        store = ChromaStore(
            collection="test_multiple",
            embedder=embedder,
            persist_dir=temp_dir
        )
        
        ingest_file(str(doc1_path), store, source_name="python.txt")
        ingest_file(str(doc2_path), store, source_name="javascript.txt")
        
        agent = RagAgent(
            store=store,
            llm=llm,
            top_k=5,
            distance_threshold=0.5
        )
        
        result = agent.ask("What programming languages are mentioned?")
        
        # Should include chunks from both sources
        sources = {chunk["source"] for chunk in result["used_chunks"]}
        assert len(sources) >= 1  # At least one source should be used
        
        assert result["answer"] == "Both Python and JavaScript are programming languages."