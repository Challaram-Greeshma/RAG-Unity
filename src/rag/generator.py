import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import LLM_MODEL, PROJECT_ROOT, UNITY_DOCS_BASE
from rag.retriever import RAGRetriever

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


class RAGGenerator:
    """Turn retrieved Unity HTML chunks into a Gemini answer with citations."""

    def __init__(
        self,
        retriever: RAGRetriever,
        model_name: str = LLM_MODEL,
        llm=None,
    ):
        self.retriever = retriever
        self.model_name = model_name
        if llm is not None:
            self.llm = llm
            return

        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY is not set. "
                "Add it to a .env file in the project root."
            )

        self.llm = ChatGoogleGenerativeAI(model=model_name)
        print(f"Loaded Gemini model: {model_name}")

    def _page_url(self, metadata: dict) -> str:
        source = metadata.get("source")
        source_path = Path(source) if source else None
        if source_path and source_path.exists():
            html = source_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r'rel="canonical" href="([^"]+)"', html)
            if match:
                return match.group(1)

        kind = metadata.get("docs_kind") or "Manual"
        source_path = str(metadata.get("source_path") or metadata.get("source_file") or "")
        normalized = source_path.replace("\\", "/")
        for prefix in ("ScriptReference/", "Manual/"):
            if prefix in normalized:
                suffix = normalized.split(prefix, 1)[1]
                folder = "ScriptReference" if prefix.startswith("Script") else "Manual"
                return f"{UNITY_DOCS_BASE}/{folder}/{suffix}"

        stem = Path(metadata.get("source_file", "")).stem
        if stem:
            return f"{UNITY_DOCS_BASE}/{kind}/{stem}.html"
        return ""

    def build_citations(self, retrieved_docs: list) -> list:
        citations = []
        for i, doc in enumerate(retrieved_docs, start=1):
            metadata = doc.get("metadata") or {}
            citations.append({
                "id": i,
                "title": metadata.get("title", ""),
                "section": metadata.get("section", ""),
                "source_file": metadata.get("source_file", "unknown"),
                "url": self._page_url(metadata),
                "score": doc.get("similarity_score", 0.0),
            })
        return citations

    def format_context(self, retrieved_docs: list) -> str:
        if not retrieved_docs:
            return ""

        parts = []
        citations = self.build_citations(retrieved_docs)
        for doc, citation in zip(retrieved_docs, citations):
            header = f"[{citation['id']}] {citation['title'] or citation['source_file']}"
            if citation["section"]:
                header += f" | Section: {citation['section']}"
            if citation["url"]:
                header += f" | {citation['url']}"
            parts.append(f"{header}\n{doc['content']}")
        return "\n\n".join(parts)

    def _as_text(self, content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful document AI assistant. Use the following context to answer the question accurately and concisely.
Cite the original HTML pages using the bracket numbers from the context, like [1] or [2].

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above, with inline citations. If the context doesn't contain enough information to answer the question, say so.""",
        )
        formatted_prompt = prompt_template.format(context=context, question=query)
        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return self._as_text(response.content)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_simple(self, query: str, context: str) -> str:
        simple_prompt = f"""Based on this context: {context}

Question: {query}

Answer:"""
        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return self._as_text(response.content)
        except Exception as e:
            return f"Error: {str(e)}"

    def answer(self, query: str, top_k: int = 5, simple: bool = False) -> str:
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        context = self.format_context(retrieved_docs)
        if not context:
            return "No relevant documents were retrieved from the vector store."
        if simple:
            return self.generate_response_simple(query, context)
        return self.generate_response(query, context)
