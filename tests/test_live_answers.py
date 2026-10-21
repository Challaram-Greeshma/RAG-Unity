import os

import pytest

from rag.config import LLM_MODEL
from rag.generator import RAGGenerator
from rag.pipeline import RAGPipeline
from rag.retriever import RAGRetriever

LIVE_QUESTIONS = [
    "What is a script",
    "How do I switch between 2D and 3D mode",
]


def _print_result(question: str, result: dict) -> None:
    print("\n" + "=" * 72)
    print(f"Q: {question}")
    print(f"Model: {LLM_MODEL}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("-" * 72)
    print(result["answer"])
    print("-" * 72)
    print("Citations:")
    for citation in result["citations"]:
        title = citation["title"] or citation["source_file"]
        section = f" — {citation['section']}" if citation["section"] else ""
        print(f"  [{citation['id']}] {title}{section} ({citation['score']:.3f})")
        if citation["url"]:
            print(f"      {citation['url']}")
    print("=" * 72 + "\n")


@pytest.fixture(scope="module")
def live_pipeline(ingested):
    if os.getenv("RUN_LLM_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_LLM_TESTS=1 to call Gemini")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY is not set")

    vector_store, embedding_manager = ingested
    retriever = RAGRetriever(vector_store, embedding_manager)
    generator = RAGGenerator(retriever)
    return RAGPipeline(retriever, generator)


@pytest.mark.llm
@pytest.mark.parametrize("question", LIVE_QUESTIONS)
def test_live_gemini_answers(live_pipeline, question):
    result = live_pipeline.query(question, top_k=5, min_score=0.2)
    _print_result(question, result)

    assert result["answer"].strip()
    assert not result["answer"].startswith("Error generating response")
    assert result["confidence"] > 0
    assert result["citations"]
    assert result["sources"]
