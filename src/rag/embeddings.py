from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        print(f"Model loaded successfully. Embedding dimension: {dim_fn()}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) >= 32,
            batch_size=EMBED_BATCH_SIZE,
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
