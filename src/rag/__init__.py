"""PDF ingest → chunk → embed → Chroma → retrieve."""

from rag.pipeline import RAGPipeline, ingest_pdfs, query
from rag.retrieve import RetrievedChunk

__all__ = ["RAGPipeline", "RetrievedChunk", "ingest_pdfs", "query"]
