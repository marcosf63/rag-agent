"""Main RAG Agent implementation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..storage.chroma_store import ChromaStore
from ..utils.logging import setup_logger
from .exceptions import AnswerNotFoundError, LLMError, RetrievalError
from .protocols import LLMProvider

log = setup_logger("rag")


@dataclass
class RagAgent:
    """
    Strict RAG Agent that only answers questions based on retrieved document context.

    Args:
        store: Vector store for document retrieval
        llm: Language model provider for answer generation
        top_k: Number of chunks to retrieve
        max_context_chars: Maximum characters to include in context
        distance_threshold: Cosine distance threshold for relevance filtering
    """

    store: ChromaStore
    llm: LLMProvider
    top_k: int = 5
    max_context_chars: int = 4000
    distance_threshold: float = 0.35

    def _format_prompt(
        self, question: str, contexts: List[Tuple[str, Dict[str, Any], float]]
    ) -> str:
        """Format the prompt with question and retrieved contexts."""
        ctx_texts = []
        total = 0
        for text, meta, dist in contexts:
            tag = f"[chunk_id={meta.get('chunk_id')} source={meta.get('source')}] (dist={round(dist, 4)})"
            addition = f"{tag}\n{text}\n"
            if total + len(addition) > self.max_context_chars:
                break
            ctx_texts.append(addition)
            total += len(addition)

        instruction = (
            "Você é um assistente **estrito** de consulta a documentos.\n"
            "- Responda SOMENTE se a resposta estiver claramente sustentada pelos trechos no CONTEXTO.\n"
            "- Cite os chunk_ids usados.\n"
            "- Se não houver informação suficiente, responda exatamente: 'Não encontrado nos documentos.'\n"
        )
        context_block = (
            "=== CONTEXTO INÍCIO ===\n" + "\n".join(ctx_texts) + "\n=== CONTEXTO FIM ==="
        )
        return f"{instruction}\n{context_block}\n\nPergunta: {question}\nResposta:"

    def ask(self, question: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question and get an answer based on retrieved documents.

        Args:
            question: The question to ask
            request_id: Optional request ID for tracking

        Returns:
            Dict containing answer, used chunks, and metadata

        Raises:
            RetrievalError: If document retrieval fails
            AnswerNotFoundError: If no relevant information is found
            LLMError: If language model generation fails
        """
        rid = request_id or str(uuid.uuid4())
        t0 = time.time()

        # Retrieval
        try:
            docs, metas, dists = self.store.query(question, k=self.top_k)
        except Exception as e:
            log.error(
                "Falha na recuperação",
                extra={"extra": {"event": "retrieval_error", "err": str(e), "rid": rid}},
            )
            raise RetrievalError(f"Falha na recuperação: {e}")

        # Filter by threshold
        triples: List[Tuple[str, Dict[str, Any], float]] = []
        for d, m, dist in zip(docs, metas, dists):
            if dist <= self.distance_threshold:
                triples.append((d, m, dist))

        if not triples:
            log.info(
                "Sem contexto relevante",
                extra={"extra": {"event": "no_context", "rid": rid, "k": self.top_k}},
            )
            raise AnswerNotFoundError("Não encontrado nos documentos.")

        # Generation
        prompt = self._format_prompt(question, triples)
        try:
            answer = self.llm.answer(prompt).strip()
        except Exception as e:
            log.error(
                "Falha no LLM", extra={"extra": {"event": "llm_error", "err": str(e), "rid": rid}}
            )
            raise LLMError(f"Falha na geração: {e}")

        # Strict adherence guardrail
        if not answer or "Não encontrado nos documentos" in answer:
            log.info(
                "Resposta negativa (guardrail ok)",
                extra={"extra": {"event": "answer_not_found", "rid": rid}},
            )
            raise AnswerNotFoundError("Não encontrado nos documentos.")

        latency = round((time.time() - t0) * 1000, 1)
        log.info(
            "Resposta gerada",
            extra={
                "extra": {
                    "event": "answer_ok",
                    "rid": rid,
                    "latency_ms": latency,
                    "used_chunks": [m.get("chunk_id") for _, m, _ in triples],
                }
            },
        )

        return {
            "request_id": rid,
            "answer": answer,
            "used_chunks": [
                {"chunk_id": m.get("chunk_id"), "distance": d, "source": m.get("source")}
                for _, m, d in triples
            ],
            "latency_ms": latency,
        }
