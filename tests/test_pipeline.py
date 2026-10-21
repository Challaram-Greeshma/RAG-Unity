from types import SimpleNamespace

from rag.generator import RAGGenerator
from rag.pipeline import RAGPipeline
from rag.retriever import RAGRetriever


class FakeLLM:
    def invoke(self, _messages):
        return SimpleNamespace(content="Scripts attach to GameObjects and inherit from MonoBehaviour [1].")


def test_build_citations_uses_unity_manual_url(ingested):
    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)
    generator = RAGGenerator(retriever, llm=FakeLLM())
    docs = retriever.retrieve("What is a script", top_k=3, score_threshold=0.2)

    citations = generator.build_citations(docs)

    assert citations
    assert citations[0]["id"] == 1
    assert citations[0]["source_file"]
    assert "docs.unity3d.com" in citations[0]["url"]
    assert citations[0]["url"].endswith(".html")


def test_query_returns_answer_sources_and_confidence(ingested):
    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)
    generator = RAGGenerator(retriever, llm=FakeLLM())
    pipeline = RAGPipeline(retriever, generator)

    result = pipeline.query("What is a script", top_k=3, min_score=0.2, return_context=True)

    assert "MonoBehaviour" in result["answer"]
    assert result["confidence"] > 0
    assert result["sources"]
    assert result["citations"]
    assert result["context"]
    assert result["sources"][0]["url"].startswith("https://docs.unity3d.com/")
    assert result["confidence"] == max(doc["score"] for doc in result["sources"])


def test_query_empty_when_nothing_passes_score_cutoff(ingested):
    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)
    generator = RAGGenerator(retriever, llm=FakeLLM())
    pipeline = RAGPipeline(retriever, generator)

    result = pipeline.query(
        "photosynthesis in rainforest canopies",
        top_k=3,
        min_score=0.85,
    )

    assert result["answer"] == "No relevant context found."
    assert result["sources"] == []
    assert result["citations"] == []
    assert result["confidence"] == 0.0
