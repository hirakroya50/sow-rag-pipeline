# PDF RAG Pipeline

A command-line tool for finding relevant passages in a collection of PDF files. It uses semantic search to retrieve matching text chunks, so you can search your documents by meaning instead of relying only on exact keywords.

This project handles PDF ingestion, embedding, vector storage, and retrieval. It returns matching passages; it does **not** generate a natural-language answer from them.

## How it works

1. Finds PDF files recursively in the configured input directory.
2. Loads each PDF page and keeps source-file metadata.
3. Splits pages into overlapping text chunks.
4. Creates embeddings with a Sentence Transformers model.
5. Stores the chunks and embeddings in a Chroma collection.
6. Embeds your query and retrieves the most similar chunks, with source names and similarity scores.

Chroma uses a local persistent store by default. You can optionally connect to Chroma Cloud.

## Technology and capabilities

- **Python 3.14+** and a `uv`-managed project
- **LangChain** PDF loading and recursive text splitting
- **Sentence Transformers** for local text embeddings (`all-MiniLM-L6-v2` by default)
- **Chroma** for local or cloud vector storage
- A CLI for indexing PDF files and retrieving ranked passages
- Configurable input directory, vector-store directory, collection, embedding model, and chunking settings

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Quickstart

Install the project and its dependencies:

```bash
uv sync
```

Put PDF files in `data/pdf/` (the default input directory), then index them:

```bash
uv run rag ingest
```

Search the indexed PDFs:

```bash
uv run rag query "What does the document say about system design?" --top-k 5
```

The query command prints each matching chunk's rank, similarity score, source file, and a short text preview.

### Ingest options

Choose another PDF directory with `--pdf-dir`. Add `--reset` to clear the configured Chroma collection before indexing:

```bash
uv run rag ingest --pdf-dir path/to/pdfs --reset
```

> **Warning:** `--reset` deletes the contents of the selected collection before adding the new PDF chunks.

## Configuration

Local Chroma is used when no Chroma Cloud API key is set; no `.env` file is required for local use.

To use Chroma Cloud, copy the example file and add the connection details from your Chroma dashboard:

```bash
cp .env.example .env
```

Set `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` in `.env`. `CHROMA_HOST` is optional and can be set when your cloud database uses a non-default host. Keep `.env` private; it is excluded from version control.

The pipeline settings can also be customized in `.env`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `RAG_PDF_DIR` | `data/pdf` | Directory searched recursively for PDF files |
| `RAG_VECTOR_DIR` | `data/vector_store` | Local Chroma persistence directory |
| `RAG_COLLECTION` | `pdf_documents` | Chroma collection name |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `RAG_CHUNK_SIZE` | `1000` | Maximum chunk size |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |

## Run tests

```bash
uv run --group dev pytest
```

## Repository name suggestion

**Recommended:** `pdf-rag-pipeline` — it describes both the PDF input and the retrieval-augmented pipeline.

Other options:

- `pdf-knowledge-search`
- `local-first-rag`
