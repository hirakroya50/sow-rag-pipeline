from langchain_core.documents import Document

from rag.ingest import load_pdfs, split_documents


def test_load_pdfs_adds_source_metadata(tmp_path, monkeypatch):
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    def fake_loader(path: str):
        class Loader:
            def load(self):
                return [Document(page_content="hello", metadata={"page": 0})]

        return Loader()

    monkeypatch.setattr("rag.ingest.PyPDFLoader", fake_loader)

    docs = load_pdfs(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["source_file"] == "notes.pdf"
    assert docs[0].metadata["file_type"] == "pdf"


def test_load_pdfs_skips_failed_file(tmp_path, monkeypatch):
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")

    def exploding_loader(_path: str):
        class Loader:
            def load(self):
                raise RuntimeError("corrupt pdf")

        return Loader()

    monkeypatch.setattr("rag.ingest.PyPDFLoader", exploding_loader)

    docs = load_pdfs(tmp_path)
    assert docs == []


def test_split_documents_creates_overlapping_chunks():
    text = ("paragraph one. " * 80) + "\n\n" + ("paragraph two. " * 80)
    docs = [Document(page_content=text, metadata={"source_file": "a.pdf"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(chunk.metadata["source_file"] == "a.pdf" for chunk in chunks)
