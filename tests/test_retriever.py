from rag.retriever import RAGRetriever


def _top_sources(docs) -> list[str]:
    return [doc["metadata"].get("source_file", "") for doc in docs]


def test_ingest_stores_chunks_with_deterministic_ids(ingested):
    vector_store, _embedding_manager = ingested
    count = vector_store.collection.count()
    assert count > 0

    sample = vector_store.collection.get(limit=min(5, count), include=["metadatas"])
    assert sample["ids"]
    assert all("::" in doc_id for doc_id in sample["ids"])


def test_retrieve_scripts_from_workflow_page(ingested):
    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)

    docs = retriever.retrieve("What is a script", top_k=5, score_threshold=0.2)

    assert docs
    assert all(doc["similarity_score"] >= 0.2 for doc in docs)
    joined = " ".join(doc["content"].lower() for doc in docs)
    assert "script" in joined
    assert "2d-game-creation-wokflow.html" in _top_sources(docs)


def test_retrieve_2d_3d_mode_switch(ingested):
    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)

    docs = retriever.retrieve(
        "How do I switch between 2D and 3D mode",
        top_k=5,
        score_threshold=0.2,
    )

    assert docs
    sources = _top_sources(docs)
    assert "2DAnd3DModeSettings.html" in sources or "2Dor3D.html" in sources
    joined = " ".join(doc["content"].lower() for doc in docs)
    assert "2d" in joined and "3d" in joined


def test_retrieve_sorting_layer(ingested):
    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)

    docs = retriever.retrieve(
        "Where is the new sorting layer",
        top_k=5,
        score_threshold=0.2,
    )

    assert docs
    assert "2d-renderer-sorting.html" in _top_sources(docs)
    joined = " ".join(doc["content"].lower() for doc in docs)
    assert "layer" in joined


def test_high_score_cutoff_filters_unrelated_query(ingested):
    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)

    docs = retriever.retrieve(
        "photosynthesis in rainforest canopies",
        top_k=5,
        score_threshold=0.85,
    )

    assert docs == []
