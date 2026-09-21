from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_pdfs(pdf_directory: str | Path) -> list[Document]:
    """Load every PDF under a directory. Skip files that fail to parse."""
    pdf_dir = Path(pdf_directory)
    documents: list[Document] = []
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    logger.info("Found %s PDF files to process", len(pdf_files))

    for pdf_file in pdf_files:
        logger.info("Processing %s", pdf_file.name)
        try:
            pages = PyPDFLoader(str(pdf_file)).load()
        except Exception:
            logger.exception("Failed to load %s", pdf_file)
            continue

        for page in pages:
            page.metadata["source_file"] = pdf_file.name
            page.metadata["file_type"] = "pdf"
        documents.extend(pages)
        logger.info("Loaded %s pages from %s", len(pages), pdf_file.name)

    logger.info("Total documents loaded: %s", len(documents))
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split pages into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %s documents into %s chunks", len(documents), len(chunks))
    return chunks
