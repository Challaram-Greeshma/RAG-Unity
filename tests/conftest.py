import shutil
from pathlib import Path

import pytest

from rag.config import DATA_DIR
from rag.pipeline import ingest

SAMPLE_FILES = (
    "2d-game-creation-wokflow.html",
    "2DAnd3DModeSettings.html",
    "2d-renderer-sorting.html",
)


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp("html")
    for name in SAMPLE_FILES:
        source = DATA_DIR / name
        if not source.exists():
            pytest.skip(f"Sample HTML file missing: {source}")
        shutil.copy(source, data_dir / name)
    return data_dir


@pytest.fixture(scope="session")
def ingested(sample_data_dir, tmp_path_factory):
    """Embed a few Unity HTML pages into a throwaway Chroma store."""
    store_path = tmp_path_factory.mktemp("chroma")
    vector_store, embedding_manager = ingest(
        data_directory=sample_data_dir,
        store_path=store_path,
        collection_name="test_html_documents",
    )
    return vector_store, embedding_manager
