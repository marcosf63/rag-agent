# RAG Agent 🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Uma biblioteca Python para RAG (Retrieval-Augmented Generation) **estrita** que responde apenas com base no contexto recuperado dos documentos indexados.

## ✨ Características

- **🎯 RAG Estrito**: Responde somente se a informação estiver claramente no contexto dos documentos
- **🔌 Provedores Plugáveis**: Suporte para embeddings e LLMs locais ou via API (OpenAI/Ollama)
- **📄 Múltiplos Formatos**: Processa arquivos TXT, MD e PDF
- **📊 Logging Estruturado**: Logs em formato JSON com rastreamento de requisições
- **📏 Filtragem por Distância**: Usa threshold de distância cosseno para relevância
- **🧪 Testado**: Suite completa de testes unitários e de integração

## 🚀 Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/marcosf63/rag-agent.git
cd rag-agent

# Instale em modo desenvolvimento
pip install -e ".[all]"
```

## 📖 Uso Básico

```python
from rag_agent import (
    RagAgent, 
    ChromaStore, 
    SentenceTransformerEmbedding,
    OllamaChat,
    ingest_file,
    AnswerNotFoundError
)

# 1. Configurar embedding (local - sem API)
embedder = SentenceTransformerEmbedding()

# 2. Criar vector store
store = ChromaStore("meus_docs", embedder)

# 3. Indexar documentos
ingest_file("documento.pdf", store)

# 4. Configurar LLM (Ollama local)
llm = OllamaChat()

# 5. Criar agente
agent = RagAgent(store=store, llm=llm)

# 6. Fazer pergunta
try:
    resultado = agent.ask("Qual procedimento está descrito no documento?")
    print(f"✅ {resultado['answer']}")
    print(f"📚 Chunks: {[c['chunk_id'] for c in resultado['used_chunks']]}")
except AnswerNotFoundError:
    print("❌ Informação não encontrada nos documentos.")
```

## 🛠️ Configuração de Provedores

### Embeddings

**Local (sem API)**:
```python
from rag_agent import SentenceTransformerEmbedding
embedder = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
```

**OpenAI** (requer `OPENAI_API_KEY`):
```python
from rag_agent import OpenAIEmbedding
embedder = OpenAIEmbedding("text-embedding-3-small")
```

### LLMs

**Ollama Local**:
```python
from rag_agent import OllamaChat
llm = OllamaChat("llama3.1:8b")  # Requer Ollama rodando
```

**OpenAI** (requer `OPENAI_API_KEY`):
```python
from rag_agent import OpenAIChat
llm = OpenAIChat("gpt-4o-mini")
```

## 📁 Estrutura do Projeto

```
rag_agent/
├── src/rag_agent/          # 📦 Código fonte da biblioteca
│   ├── core/               # 🧠 Componentes principais
│   ├── providers/          # 🔌 Provedores de embedding/LLM  
│   ├── storage/            # 💾 Armazenamento vetorial
│   └── utils/              # 🛠️ Utilitários
├── tests/                  # 🧪 Testes (unitários + integração)
├── examples/               # 📚 Exemplos de uso
├── config/                 # ⚙️ Configurações
└── docs/                   # 📖 Documentação
```

## 🔧 Comandos Úteis

```bash
# Testes
make test              # Todos os testes
make test-unit         # Apenas testes unitários
make test-integration  # Apenas testes de integração

# Qualidade do código
make lint              # Linting
make format            # Formatação automática
make type-check        # Verificação de tipos

# Desenvolvimento
make install-dev       # Instalar com deps de desenvolvimento
make run-example       # Executar exemplo básico
```

## 📊 Estrutura de Resposta

```json
{
  "request_id": "uuid-da-requisição",
  "answer": "Resposta baseada no contexto recuperado",
  "used_chunks": [
    {
      "chunk_id": 0,
      "distance": 0.15,
      "source": "documento.pdf"
    }
  ],
  "latency_ms": 1250.5
}
```

## ⚙️ Parâmetros de Configuração

| Parâmetro | Padrão | Descrição |
|-----------|---------|-----------|
| `top_k` | 5 | Número de chunks recuperados |
| `max_context_chars` | 4000 | Tamanho máximo do contexto |
| `distance_threshold` | 0.35 | Threshold de distância cosseno |
| `max_chars` | 1200 | Tamanho dos chunks |
| `overlap` | 120 | Sobreposição entre chunks |

## 🚨 Tratamento de Erros

- `AnswerNotFoundError`: Informação não encontrada nos documentos
- `RetrievalError`: Falha na busca vetorial  
- `LLMError`: Falha na geração de resposta
- `EmbeddingError`: Falha na geração de embeddings
- `IngestionError`: Falha no processamento de documentos

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para mais detalhes.

## 📄 Licença

Este projeto está sob a licença MIT. Veja [LICENSE](LICENSE) para mais detalhes.

## 🎯 Exemplo Completo

Confira `examples/basic_usage.py` para um exemplo completo de uso da biblioteca.