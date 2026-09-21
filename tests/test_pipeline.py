from langchain_core.documents import Document
import numpy as np

from rag.pipeline import RAGPipeline


class FakeEmbedder:
    def generate_embeddings(self, texts):
        return np.ones((len(texts), 3))


class FakeStore:
    def __init__(self):
        self.added = []
        self.collection = self
        self.reset_called = False

    def reset(self):
        self.reset_called = True
        self.added = []

    def add_documents(self, documents, embeddings):
        self.added.append((list(documents), embeddings))

    def query(self, query_embeddings, n_results):
        return {
            "ids": [["1"]],
            "documents": [["chunk"]],
            "metadatas": [[{}]],
            "distances": [[0.0]],
        }


def test_ingest_pdfs_reset_clears_then_adds(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    def fake_loader(_path: str):
        class Loader:
            def load(self):
                return [Document(page_content="short text", metadata={"page": 0})]

        return Loader()

    monkeypatch.setattr("rag.ingest.PyPDFLoader", fake_loader)

    store = FakeStore()
    pipeline = RAGPipeline(embedder=FakeEmbedder(), store=store)
    count = pipeline.ingest_pdfs(tmp_path, reset=True)

    assert store.reset_called is True
    assert count >= 1
    assert len(store.added) == 1


def test_query_uses_retriever():
    pipeline = RAGPipeline(embedder=FakeEmbedder(), store=FakeStore())
    hits = pipeline.query("anything", top_k=1)
    assert hits[0].content == "chunk"
