import json
from unittest.mock import MagicMock, patch

import pytest
from schemas import StatementVerdict, Verdict
from tools import (
    _cache_key,
    _expansion_cache_key,
    classify_support,
    expand_query,
    reciprocal_rank_fusion,
    rerank,
)


@pytest.mark.unit
class TestTools:
    def test_reciprocal_rank_fusion_basic(self):
        c1 = {"chunk_id": "c1", "text": "text 1"}
        c2 = {"chunk_id": "c2", "text": "text 2"}
        c3 = {"chunk_id": "c3", "text": "text 3"}

        list1 = [c1, c2]
        list2 = [c2, c3]

        fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=10)
        assert len(fused) == 3
        top_chunk, top_score = fused[0]
        assert top_chunk["chunk_id"] == "c2"

        expected_score_c2 = (1.0 / 62) + (1.0 / 61)
        assert pytest.approx(top_score, rel=1e-4) == expected_score_c2

    def test_reciprocal_rank_fusion_weights(self):
        c1 = {"chunk_id": "c1"}
        c2 = {"chunk_id": "c2"}

        list1 = [c1]
        list2 = [c2]

        fused = reciprocal_rank_fusion([list1, list2], weights=[2.0, 1.0], k=60)
        assert fused[0][0]["chunk_id"] == "c1"
        assert pytest.approx(fused[0][1], rel=1e-4) == 2.0 / 61
        assert pytest.approx(fused[1][1], rel=1e-4) == 1.0 / 61

    def test_reciprocal_rank_fusion_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_cache_keys_deterministic(self, sample_chunks):
        k1 = _expansion_cache_key("Mojżesz wyprowadził Izraelitów")
        k2 = _expansion_cache_key("  mojżesz wyprowadził izraelitów  ")
        assert k1 == k2

        ck1 = _cache_key("Mojżesz", sample_chunks)
        ck2 = _cache_key("Mojżesz", list(reversed(sample_chunks)))
        assert ck1 == ck2

    def test_rerank_fallback_without_cross_encoder(self, sample_chunks):
        scores = [0.1, 0.9, 0.5]
        reranked = rerank("query", sample_chunks, scores=scores)
        assert len(reranked) == len(sample_chunks)
        assert reranked[0][0]["chunk_id"] == sample_chunks[1]["chunk_id"]
        assert reranked[0][1] == 0.9

    @patch("tools._get_client")
    def test_expand_query_with_mock_llm(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "queries": [
                    "Mojżesz rzekł do ludu swego",
                    "Wyjście synów Izraela z ziemi egipskiej",
                ]
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        statement = "Mojżesz wyprowadził Żydów z Egiptu"
        queries = expand_query(statement, use_cache=False)

        assert len(queries) >= 2
        assert statement in queries
        assert "Mojżesz rzekł do ludu swego" in queries

    @patch("tools._get_client")
    def test_classify_support_with_mock_llm(self, mock_get_client, sample_chunks):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "verdict": "directly_supported",
                "confidence": 0.95,
                "citations": [
                    {
                        "ref": "Wj 3:10",
                        "quote": "Idź przeto teraz, oto posyłam cię do faraona, i wyprowadź mój lud, Izraelitów, z Egiptu.",
                        "relation": "supports",
                    }
                ],
                "reasoning": "Tekst wprost nakazuje Mojżeszowi wyprowadzić lud z Egiptu.",
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        verdict = classify_support(
            "Mojżesz wyprowadził Izraelitów",
            sample_chunks,
            use_cache=False,
        )

        assert isinstance(verdict, StatementVerdict)
        assert verdict.verdict == Verdict.SUPPORTED
        assert verdict.confidence == 0.95
        assert len(verdict.citations) == 1
        assert verdict.citations[0].ref == "Wj 3:10"
