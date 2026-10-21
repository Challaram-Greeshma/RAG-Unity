from pathlib import Path

from rag.chunker import chunk_html_file
from rag.config import (
    COLLECTION_NAME,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    INGEST_FLUSH_SIZE,
    VECTOR_STORE_DIR,
    default_ingest_dir,
)
from rag.data_loader import iter_html_files
from rag.embeddings import EmbeddingManager
from rag.generator import RAGGenerator
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStore


class RAGPipeline:
    """Retrieve Unity HTML chunks from ChromaDB and pass them to Gemini as context."""

    def __init__(self, retriever: RAGRetriever, generator: RAGGenerator):
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        return_context: bool = False,
    ) -> dict:
        print("=== RAG Pipeline ===")
        print(f"Question: {question}")
        print(f"Top K: {top_k}, Min score: {min_score}")

        retrieved_docs = self.retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=min_score,
        )

        if not retrieved_docs:
            output = {
                "answer": "No relevant context found.",
                "sources": [],
                "citations": [],
                "confidence": 0.0,
            }
            if return_context:
                output["context"] = ""
            print(output["answer"])
            return output

        context = self.generator.format_context(retrieved_docs)
        citations = self.generator.build_citations(retrieved_docs)
        sources = []
        for citation, doc in zip(citations, retrieved_docs):
            preview = doc["content"][:300]
            if len(doc["content"]) > 300:
                preview += "..."
            sources.append({
                "id": citation["id"],
                "source": citation["source_file"],
                "title": citation["title"],
                "section": citation["section"],
                "url": citation["url"],
                "score": citation["score"],
                "preview": preview,
            })
        confidence = max(doc["similarity_score"] for doc in retrieved_docs)

        print(f"Sending {len(retrieved_docs)} retrieved chunks to Gemini...")
        answer = self.generator.generate_response(question, context)

        print("\n--- Answer ---")
        print(answer)
        print(f"\n--- Confidence: {confidence:.3f} ---")
        print("\n--- Citations ---")
        for citation in citations:
            title = citation["title"] or citation["source_file"]
            section = f" — {citation['section']}" if citation["section"] else ""
            print(f"[{citation['id']}] {title}{section}")
            if citation["url"]:
                print(f"    {citation['url']}")

        output = {
            "answer": answer,
            "sources": sources,
            "citations": citations,
            "confidence": confidence,
        }
        if return_context:
            output["context"] = context
        return output


def _flush_chunks(vector_store: VectorStore, embedding_manager: EmbeddingManager, chunks):
    embeddings = embedding_manager.generate_embeddings(
        [chunk.page_content for chunk in chunks]
    )
    vector_store.add_documents(chunks, embeddings)


def ingest(
    data_directory: str | Path | None = None,
    store_path: str | Path | None = None,
    collection_name: str | None = None,
    scope: str = "all",
) -> tuple[VectorStore, EmbeddingManager]:
    """Load HTML, chunk, embed, and write to ChromaDB in batches."""
    data_directory = Path(data_directory or default_ingest_dir())
    html_files = list(iter_html_files(data_directory, scope=scope))
    print(f"Ingesting {len(html_files)} HTML files from {data_directory} (scope={scope})")

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(
        embedding_manager,
        store_path=store_path or VECTOR_STORE_DIR,
        collection_name=collection_name or COLLECTION_NAME,
    )
    vector_store.reset()

    pending = []
    total_chunks = 0
    errors = 0
    for index, html_file in enumerate(html_files, start=1):
        try:
            pending.extend(chunk_html_file(html_file, data_directory))
        except Exception as e:
            errors += 1
            if errors <= 20:
                print(f"  [error] {html_file}: {e}")
            continue

        if index % 100 == 0 or index == len(html_files):
            print(
                f"Parsed {index}/{len(html_files)} pages "
                f"({total_chunks + len(pending)} chunks, {errors} errors)"
            )

        if len(pending) >= INGEST_FLUSH_SIZE:
            _flush_chunks(vector_store, embedding_manager, pending)
            total_chunks += len(pending)
            pending = []
            print(f"Stored {total_chunks} chunks")

    if pending:
        _flush_chunks(vector_store, embedding_manager, pending)
        total_chunks += len(pending)

    print(f"Total documents in store: {vector_store.collection.count()}")
    print(
        f"Ingest complete: {total_chunks} chunks from {len(html_files)} files "
        f"({errors} errors)"
    )
    return vector_store, embedding_manager


def build_pipeline(
    embedding_manager: EmbeddingManager | None = None,
    vector_store: VectorStore | None = None,
) -> RAGPipeline:
    """Open the existing vector store and return a queryable RAG pipeline."""
    embedding_manager = embedding_manager or EmbeddingManager()
    vector_store = vector_store or VectorStore(embedding_manager)
    retriever = RAGRetriever(vector_store, embedding_manager)
    generator = RAGGenerator(retriever)
    return RAGPipeline(retriever, generator)
