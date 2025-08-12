"""Document ingestion utilities."""

import os
from typing import Optional

from ..core.exceptions import IngestionError
from ..storage.chroma_store import ChromaStore
from .logging import setup_logger
from .text_processing import chunk_text

log = setup_logger("rag")


def read_text_from_path(path: str) -> str:
    """
    Read text content from various file formats.

    Args:
        path: Path to the file

    Returns:
        Extracted text content

    Raises:
        IngestionError: If file reading fails
    """
    if not os.path.exists(path):
        raise IngestionError(f"Arquivo não encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            try:
                import pypdf  # pip install pypdf
            except ImportError as e:
                raise IngestionError(f"Para PDF, instale pypdf: {e}")
            reader = pypdf.PdfReader(path)
            texts = []
            for i, page in enumerate(reader.pages):
                t = page.extract_text() or ""
                texts.append(t)
            return "\n".join(texts)
        else:
            # Simple fallback: try to open as text
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        raise IngestionError(f"Falha lendo {path}: {e}")


def ingest_file(
    path: str,
    store: ChromaStore,
    source_name: Optional[str] = None,
    max_chars: int = 1200,
    overlap: int = 120,
):
    """
    Ingest a file into the vector store.

    Args:
        path: Path to the file to ingest
        store: Vector store instance
        source_name: Optional source name override
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks

    Raises:
        IngestionError: If ingestion fails
    """
    try:
        text = read_text_from_path(path)
        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        metadatas = [
            {"source": source_name or os.path.basename(path), "chunk_id": i}
            for i in range(len(chunks))
        ]
        store.upsert(chunks, metadatas)
        log.info(
            "Ingestão concluída",
            extra={
                "extra": {
                    "event": "ingest_ok",
                    "source": source_name or os.path.basename(path),
                    "chunks": len(chunks),
                }
            },
        )
    except Exception as e:
        log.error(
            "Falha na ingestão",
            extra={"extra": {"event": "ingest_error", "err": str(e), "source": path}},
        )
        raise IngestionError(str(e))
