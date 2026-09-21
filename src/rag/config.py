from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Settings:
    pdf_dir: Path = _path_from_env("RAG_PDF_DIR", PROJECT_ROOT / "data" / "pdf")
    vector_dir: Path = _path_from_env("RAG_VECTOR_DIR", PROJECT_ROOT / "data" / "vector_store")
    collection_name: str = os.getenv("RAG_COLLECTION", "pdf_documents")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
