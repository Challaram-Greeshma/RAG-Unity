from pathlib import Path
from typing import List

import chromadb
import numpy as np
from langchain_core.documents import Document

from rag.config import COLLECTION_NAME, VECTOR_STORE_DIR
from rag.embeddings import EmbeddingManager


class VectorStore:
    """Stores document chunks and embeddings in a persistent ChromaDB collection."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        store_path: str | Path = VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.embedding_manager = embedding_manager
        self.store_path = str(store_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        Path(self.store_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.store_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store initialized at {self.store_path}")
        print(f"Collection: {self.collection_name}")
        print(f"Existing documents: {self.collection.count()}")

    def _clean_metadata(self, metadata: dict) -> dict:
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    def reset(self):
        count = self.collection.count() if self.collection is not None else 0
        if count:
            print(f"Resetting collection ({count} existing docs) to avoid duplicates")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            source_path = doc.metadata.get("source_path") or doc.metadata.get(
                "source_file", "document"
            )
            block_index = doc.metadata.get("block_index", i)
            ids.append(f"{source_path}::{block_index}")

            metadata = self._clean_metadata(dict(doc.metadata))
            metadata["doc_index"] = i
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents_text,
            metadatas=metadatas,
        )

    def query(self, query_text: str, n_results: int = 5):
        query_embedding = self.embedding_manager.generate_embeddings([query_text])
        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
