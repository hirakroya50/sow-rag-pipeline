import numpy as np

from rag.retrieve import RAGRetriever, hits_from_query_results


class FakeStore:
    def __init__(self, results):
        self.collection = self
        self._results = results

    def query(self, query_embeddings, n_results):
        return self._results


class FakeEmbedder:
    def generate_embeddings(self, texts):
        return np.array([[0.1, 0.2] for _ in texts])


def test_hits_from_query_results_filters_by_score():
    results = {
        "ids": [["a", "b"]],
        "documents": [["keep me", "drop me"]],
        "metadatas": [[{"page": 1}, {"page": 2}]],
        "distances": [[0.1, 0.9]],
    }
    hits = hits_from_query_results(results, score_threshold=0.5)
    assert len(hits) == 1
    assert hits[0].id == "a"
    assert hits[0].similarity_score == 0.9
    assert hits[0].rank == 1


def test_retriever_returns_ranked_hits():
    store = FakeStore(
        {
            "ids": [["doc-1"]],
            "documents": [["LLD parking lot"]],
            "metadatas": [[{"source_file": "system-design.pdf"}]],
            "distances": [[0.2]],
        }
    )
    retriever = RAGRetriever(vector_store=store, embedding_manager=FakeEmbedder())
    hits = retriever.retrieve("parking lot", top_k=1)
    assert len(hits) == 1
    assert hits[0].content == "LLD parking lot"
    assert hits[0].metadata["source_file"] == "system-design.pdf"
