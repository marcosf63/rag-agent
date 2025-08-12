"""ChromaDB vector store implementation."""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..core.protocols import EmbeddingProvider


class ChromaStore:
    """Vector store wrapper using ChromaDB for document storage and similarity search."""
    
    def __init__(self, collection: str, embedder: EmbeddingProvider, persist_dir: str = "./chroma_db"):
        import chromadb
        from chromadb.config import Settings
        
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=persist_dir, 
            settings=Settings(allow_reset=False)
        )
        # NOTE: We pass embeddings manually in upserts/queries
        self.col = self.client.get_or_create_collection(
            name=collection, 
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Insert or update documents in the vector store."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        vectors = self.embedder.embed(texts)
        self.col.upsert(documents=texts, metadatas=metadatas, embeddings=vectors, ids=ids)

    def query(self, text: str, k: int = 5) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Query the vector store for similar documents."""
        vec = self.embedder.embed([text])[0]
        res = self.col.query(
            query_embeddings=[vec], 
            n_results=k, 
            include=["documents", "metadatas", "distances"]
        )
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        return docs, metas, dists