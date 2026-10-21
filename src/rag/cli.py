import argparse

from dotenv import load_dotenv

from rag.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, PROJECT_ROOT, default_ingest_dir
from rag.pipeline import build_pipeline, ingest


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Unity docs RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load HTML into the vector store")
    ingest_parser.add_argument(
        "--data",
        default=str(default_ingest_dir()),
        help="Directory of HTML files (defaults to Documentation/en when present)",
    )
    ingest_parser.add_argument(
        "--scope",
        choices=["all", "manual", "script"],
        default="all",
        help="Index Manual, Scripting API, or both",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the vector store")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ask_parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.data, scope=args.scope)
        return

    pipeline = build_pipeline()
    pipeline.query(args.question, top_k=args.top_k, min_score=args.min_score)
