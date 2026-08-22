import pytest
from bm_25 import (
    bm25_retrieve,
    get_bm25_index,
    multi_bm25_retrieve,
    tokenize_text,
)


@pytest.mark.unit
class TestBM25:
    def test_tokenize_text_removes_stopwords(self):
        text = "i oraz ale Bóg stworzył niebo"
        tokens = tokenize_text(text)
        assert "i" not in tokens
        assert "oraz" not in tokens
        assert "ale" not in tokens
        assert "bóg" in tokens or any("bóg" in t or "bog" in t for t in tokens)
        assert "niebo" in tokens

    def test_tokenize_text_empty_and_punctuation(self):
        assert tokenize_text("") == []
        assert tokenize_text("   ... ,,, !!! ??? ") == []
        assert tokenize_text("a w z") == []

    def test_tokenize_text_lemmatization(self):
        text = "apostołowie poszli do Jerozolimy"
        tokens = tokenize_text(text)
        assert len(tokens) > 0
        assert any("apostoł" in t for t in tokens)

    def test_get_bm25_index_loads(self):
        index, chunks = get_bm25_index()
        assert index is not None
        assert chunks is not None
        assert len(chunks) > 0
        assert "chunk_id" in chunks[0]
        assert "text" in chunks[0]

    def test_bm25_retrieve_basic(self):
        query = "stworzenie nieba i ziemi"
        results = bm25_retrieve(query, top_k=3)
        assert len(results) <= 3
        if results:
            chunk, score = results[0]
            assert "chunk_id" in chunk
            assert isinstance(score, float)
            assert score > 0

    def test_bm25_retrieve_empty_query(self):
        results = bm25_retrieve("", top_k=5)
        assert results == []

    def test_multi_bm25_retrieve(self):
        queries = ["Mojżesz Egipt", "Jezus wino Kana"]
        results = multi_bm25_retrieve(queries, top_k_per_query=5)
        assert isinstance(results, list)
        if results:
            first_chunk, score = results[0]
            assert "chunk_id" in first_chunk
            assert isinstance(score, float)

    def test_multi_bm25_retrieve_empty_list(self):
        results = multi_bm25_retrieve([])
        assert results == []
