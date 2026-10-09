from __future__ import annotations

import logging
from pathlib import Path

from rag.config import Settings
from rag.embeddings import EmbeddingManager
from rag.ingest import load_pdfs, split_documents
from rag.retrieve import RAGRetriever, RetrievedChunk
from rag.store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Load PDFs, index chunks, and retrieve similar passages."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingManager | None = None,
        store: VectorStore | None = None,
    ):
        self.settings = settings or Settings()
        self._embedder = embedder
        self._store = store

    @property
    def embedder(self) -> EmbeddingManager:
        if self._embedder is None:
            self._embedder = EmbeddingManager(self.settings.embedding_model)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(
                collection_name=self.settings.collection_name,
                persist_directory=self.settings.vector_dir,
                chroma_api_key=self.settings.chroma_api_key,
                chroma_tenant=self.settings.chroma_tenant,
                chroma_database=self.settings.chroma_database,
                chroma_host=self.settings.chroma_host,
            )
        return self._store

    def ingest_pdfs(self, pdf_dir: str | Path | None = None, *, reset: bool = False) -> int:
        directory = Path(pdf_dir) if pdf_dir is not None else self.settings.pdf_dir
        if not directory.exists():
            raise FileNotFoundError(f"PDF directory does not exist: {directory}")

        documents = load_pdfs(directory)
        chunks = split_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            logger.warning("No chunks produced from %s", directory)
            return 0

        if reset:
            self.store.reset()

        embeddings = self.embedder.generate_embeddings([chunk.page_content for chunk in chunks])
        self.store.add_documents(chunks, embeddings)
        return len(chunks)

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        retriever = RAGRetriever(self.store, self.embedder)
        return retriever.retrieve(text, top_k=top_k, score_threshold=score_threshold)


def ingest_pdfs(pdf_dir: str | Path | None = None, *, reset: bool = False) -> int:
    return RAGPipeline().ingest_pdfs(pdf_dir, reset=reset)


def query(text: str, *, top_k: int = 5, score_threshold: float = 0.0) -> list[RetrievedChunk]:
    return RAGPipeline().query(text, top_k=top_k, score_threshold=score_threshold)
