from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    content: str
    metadata: dict[str, Any]
    similarity_score: float
    distance: float
    rank: int


def hits_from_query_results(
    results: dict[str, Any],
    score_threshold: float = 0.0,
) -> list[RetrievedChunk]:
    if not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]
    hits: list[RetrievedChunk] = []

    for i, (doc_id, content, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True)
    ):
        similarity_score = 1 - distance
        if similarity_score < score_threshold:
            continue
        hits.append(
            RetrievedChunk(
                id=doc_id,
                content=content,
                metadata=metadata or {},
                similarity_score=similarity_score,
                distance=distance,
                rank=i + 1,
            )
        )
    return hits


class RAGRetriever:
    def __init__(self, vector_store: Any, embedding_manager: Any):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        logger.info("Retrieving top_k=%s threshold=%s query=%r", top_k, score_threshold, query)
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        hits = hits_from_query_results(results, score_threshold=score_threshold)
        logger.info("Retrieved %s chunks", len(hits))
        return hits
