from __future__ import annotations

import argparse
import logging
import sys

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="PDF RAG ingest and retrieve")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load PDFs, chunk, embed, and store")
    ingest.add_argument("--pdf-dir", default=None, help="Directory of PDF files")
    ingest.add_argument("--reset", action="store_true", help="Clear the collection before ingest")

    query = sub.add_parser("query", help="Retrieve similar chunks")
    query.add_argument("text", help="Search query")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--score-threshold", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    if args.command == "ingest":
        count = pipeline.ingest_pdfs(args.pdf_dir, reset=args.reset)
        print(f"Indexed {count} chunks")
        return 0

    hits = pipeline.query(args.text, top_k=args.top_k, score_threshold=args.score_threshold)
    if not hits:
        print("No matching chunks")
        return 0
    for hit in hits:
        source = hit.metadata.get("source_file", "")
        print(f"[{hit.rank}] score={hit.similarity_score:.3f} {source}")
        print(hit.content[:400].replace("\n", " "))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
