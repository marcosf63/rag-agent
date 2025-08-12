"""Embedding provider implementations."""

from typing import List

from ..core.exceptions import EmbeddingError


class OpenAIEmbedding:
    """OpenAI embedding provider using text-embedding-3-small by default."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}")


class SentenceTransformerEmbedding:
    """Local embedding provider using SentenceTransformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(f"sentence-transformers não instalado: {e}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local SentenceTransformer model."""
        try:
            return self.model.encode(
                texts, convert_to_numpy=False, normalize_embeddings=True
            ).tolist()
        except Exception as e:
            raise EmbeddingError(f"ST embedding failed: {e}")
