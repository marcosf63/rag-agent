"""Protocol definitions for pluggable providers."""

from typing import List, Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per text)
        """
        ...


class LLMProvider(Protocol):
    """Protocol for language model providers."""
    
    def answer(self, prompt: str) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated response text
        """
        ...