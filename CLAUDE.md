# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) agent implementation in Python that provides strict document-based question answering. The agent only responds with information directly supported by the ingested document context.

## Architecture

### Core Components

- **RagAgent**: Main agent class that handles question answering with strict adherence to document context
- **ChromaStore**: Vector database wrapper using ChromaDB for document storage and similarity search
- **Embedding Providers**: Pluggable interface supporting OpenAI embeddings and local SentenceTransformers
- **LLM Providers**: Pluggable interface supporting OpenAI Chat and local Ollama models
- **Document Ingestion**: Handles text extraction from various formats (TXT, MD, PDF) and chunking

### Key Features

- **Strict RAG**: Only answers questions when information is clearly present in the document context
- **Pluggable Providers**: Modular design for embeddings and LLMs (cloud or local)
- **Distance Filtering**: Uses cosine distance threshold (default 0.35) to filter relevant chunks
- **JSON Logging**: Structured logging with request tracking
- **Error Handling**: Custom exception hierarchy for different failure modes

## Dependencies

Install based on your chosen providers:
- Base: `pip install chromadb`
- OpenAI: `pip install openai`
- Local embeddings: `pip install sentence-transformers`
- PDF support: `pip install pypdf`
- Ollama: Install and run Ollama locally

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI embedding/LLM providers

## Running the Agent

The main file `rag_agent.py` includes a complete example at the bottom. To run:

1. Configure your embedding provider (OpenAI or SentenceTransformers)
2. Set up ChromaDB collection
3. Ingest documents using `ingest_file()`
4. Initialize RagAgent with your LLM provider
5. Call `agent.ask(question)` for queries

## Key Configuration Parameters

- `top_k`: Number of chunks to retrieve (default: 5)
- `max_context_chars`: Maximum context size for LLM (default: 4000)
- `distance_threshold`: Cosine distance threshold for relevance (default: 0.35)
- `max_chars`: Chunk size for document splitting (default: 1200)
- `overlap`: Character overlap between chunks (default: 120)

## Error Handling

The system uses a hierarchy of custom exceptions:
- `AnswerNotFoundError`: No relevant information found in documents
- `RetrievalError`: Vector search failures
- `LLMError`: Language model generation failures
- `EmbeddingError`: Embedding generation failures
- `IngestionError`: Document processing failures