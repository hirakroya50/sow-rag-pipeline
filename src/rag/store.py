from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[str(key)] = value
        else:
            clean[str(key)] = str(value)
    return clean


def document_id(document: Document, index: int) -> str:
    source = str(document.metadata.get("source") or document.metadata.get("source_file") or "")
    page = str(document.metadata.get("page", ""))
    digest = hashlib.sha256(f"{source}:{page}:{index}:{document.page_content[:120]}".encode()).hexdigest()
    return f"doc_{digest[:12]}_{index}"


class VectorStore:
    """Persistent Chroma collection for PDF chunk embeddings."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str | Path = "data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_or_create_collection()
        logger.info(
            "Vector store ready. Collection=%s count=%s",
            self.collection_name,
            self.collection.count(),
        )

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Reset collection %s", self.collection_name)

    def add_documents(self, documents: list[Document], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        if not documents:
            return

        ids = [document_id(doc, i) for i, doc in enumerate(documents)]
        metadatas = []
        for i, doc in enumerate(documents):
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(_sanitize_metadata(metadata))

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=[doc.page_content for doc in documents],
        )
        logger.info(
            "Upserted %s chunks. Collection count=%s",
            len(documents),
            self.collection.count(),
        )
