from rag.config import Settings


def test_settings_use_notebook_defaults():
    settings = Settings()
    assert settings.collection_name == "pdf_documents"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
