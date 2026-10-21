from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "Documentation" / "en"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = "html_documents"
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
LLM_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.2
EMBED_BATCH_SIZE = 64
INGEST_FLUSH_SIZE = 500
UNITY_DOCS_BASE = "https://docs.unity3d.com/6000.7/Documentation"


def default_ingest_dir() -> Path:
    """Prefer the full Unity dump when it is present; otherwise the sample data/ HTML."""
    if (DOCS_DIR / "Manual").is_dir() or (DOCS_DIR / "ScriptReference").is_dir():
        return DOCS_DIR
    return DATA_DIR
