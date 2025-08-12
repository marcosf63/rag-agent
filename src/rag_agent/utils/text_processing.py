"""Text processing utilities."""

from typing import List

from ..core.exceptions import IngestionError


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks

    Raises:
        IngestionError: If max_chars <= overlap
    """
    if max_chars <= overlap:
        raise IngestionError("max_chars precisa ser maior que overlap.")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks
