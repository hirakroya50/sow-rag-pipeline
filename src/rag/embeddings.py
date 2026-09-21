from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Generate sentence embeddings with a local SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        dimension = getattr(self.model, "get_embedding_dimension", None)
        if callable(dimension):
            size = dimension()
        else:
            size = self.model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (dim=%s)", model_name, size)

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        logger.info("Generating embeddings for %s texts", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings)
