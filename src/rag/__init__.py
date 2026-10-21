from rag.chunker import split_documents
from rag.data_loader import load_html_file, process_all_data
from rag.embeddings import EmbeddingManager
from rag.generator import RAGGenerator
from rag.pipeline import RAGPipeline, build_pipeline, ingest
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStore


def main() -> None:
    from rag.cli import main as cli_main

    cli_main()


__all__ = [
    "EmbeddingManager",
    "RAGGenerator",
    "RAGPipeline",
    "RAGRetriever",
    "VectorStore",
    "build_pipeline",
    "ingest",
    "load_html_file",
    "main",
    "process_all_data",
    "split_documents",
]
