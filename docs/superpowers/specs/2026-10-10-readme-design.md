# README Design: PDF RAG Pipeline

## Goal

Replace the empty repository README with a concise, practical guide that explains
the problem this project addresses, its PDF retrieval workflow, its technical
stack, and how to install and use the CLI.

## Audience and scope

The audience is a developer evaluating or running this repository. The README
will cover the existing PDF-only workflow and avoid implying support for other
document types or features that are not implemented. "Skills used" means the
project's technologies and capabilities, not AI-assistant skills.

## Content and usage flow

The README will introduce the project as a PDF semantic-search/RAG pipeline,
then explain the flow: discover PDFs, load pages, split into overlapping text
chunks, create sentence embeddings, store vectors in Chroma, and retrieve
ranked matches for a query. It will describe local Chroma as the default and
Chroma Cloud as an optional configuration.

The quickstart will require Python 3.14 or newer and `uv`, show dependency
installation, optional `.env` setup, PDF ingestion, querying, and tests. CLI
examples and configuration names will be checked against `pyproject.toml`,
`.env.example`, and the implementation.

## Repository name recommendation

Recommend `pdf-rag-pipeline` as the clearest name for the repository. Mention
alternatives only as suggestions; do not rename the GitHub repository or change
its remote.

## Validation

Review the rendered Markdown structure and run the existing test suite. Ensure
every documented command and configuration variable matches the checked-in
implementation.
