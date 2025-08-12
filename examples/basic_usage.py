#!/usr/bin/env python3
"""
Exemplo básico de uso do RAG Agent.
"""

import json
from pathlib import Path
import sys

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent import (
    RagAgent,
    ChromaStore,
    SentenceTransformerEmbedding,
    OllamaChat,
    OpenAIEmbedding,
    OpenAIChat,
    ingest_file,
    AnswerNotFoundError,
    RagError,
)


def main():
    """Exemplo completo de uso do RAG Agent."""
    
    # 1) Escolha o provedor de embeddings
    # Para usar OpenAI (requer OPENAI_API_KEY):
    # embedder = OpenAIEmbedding()
    
    # Para usar modelo local:
    embedder = SentenceTransformerEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    # 2) Banco vetorial (persistência em ./chroma_db)
    store = ChromaStore(
        collection="exemplo_basico", 
        embedder=embedder, 
        persist_dir="./chroma_db"
    )

    # 3) Ingestão (faça 1x ou quando atualizar os arquivos)
    # Descomente e ajuste os caminhos conforme necessário:
    # ingest_file("exemplo.txt", store, source_name="exemplo.txt")
    # ingest_file("manual.pdf", store, source_name="manual.pdf")

    # 4) LLM (escolha um)
    # Para usar OpenAI:
    # llm = OpenAIChat("gpt-4o-mini")
    
    # Para usar Ollama local:
    llm = OllamaChat(model="llama3.1:8b")

    # 5) Criar o agente
    agent = RagAgent(
        store=store, 
        llm=llm, 
        top_k=5, 
        distance_threshold=0.35
    )

    # 6) Fazer perguntas
    questions = [
        "Qual é o procedimento recomendado na seção inicial?",
        "Como configurar o sistema?",
        "Quais são os requisitos principais?",
    ]

    for question in questions:
        print(f"\n🤔 Pergunta: {question}")
        try:
            result = agent.ask(question)
            print(f"✅ Resposta: {result['answer']}")
            print(f"📊 Chunks utilizados: {[c['chunk_id'] for c in result['used_chunks']]}")
            print(f"⏱️  Latência: {result['latency_ms']}ms")
        except AnswerNotFoundError:
            print("❌ Não encontrado nos documentos.")
        except RagError as e:
            print(f"⚠️  Erro: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()