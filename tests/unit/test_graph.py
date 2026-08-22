from unittest.mock import patch

import pytest
from graph import (
    build_graph,
    check_relevance,
    classify_node,
    expand_query_node,
    verify_statement,
)
from schemas import Verdict


@pytest.mark.unit
class TestGraph:
    def test_check_relevance_broaden_when_low_score(self):
        state = {
            "statement": "Test query",
            "rerank_scores": [0.1, 0.2],
            "retrieval_attempts": 1,
        }
        decision = check_relevance(state)
        assert decision == "broaden"

    def test_check_relevance_classify_when_high_score(self):
        state = {
            "statement": "Test query",
            "rerank_scores": [0.8, 0.2],
            "retrieval_attempts": 1,
        }
        decision = check_relevance(state)
        assert decision == "classify"

    def test_check_relevance_classify_when_max_attempts_reached(self):
        state = {
            "statement": "Test query",
            "rerank_scores": [0.1, 0.2],
            "retrieval_attempts": 2,
        }
        decision = check_relevance(state)
        assert decision == "classify"

    @patch("graph.expand_query")
    def test_expand_query_node(self, mock_expand):
        mock_expand.return_value = ["q1", "q2"]
        state = {"statement": "original", "expanded_queries": []}
        new_state = expand_query_node(state)
        assert new_state["expanded_queries"] == ["q1", "q2"]

    @patch("graph.classify_support")
    def test_classify_node(self, mock_classify, sample_verdict, sample_chunks):
        mock_classify.return_value = sample_verdict
        state = {
            "statement": "Mojżesz wyprowadził Izraelitów",
            "chunks": sample_chunks,
            "verdict": None,
        }
        new_state = classify_node(state)
        assert new_state["verdict"] == sample_verdict

    def test_build_graph_compiles(self):
        app = build_graph()
        assert app is not None

    @patch("graph.expand_query")
    @patch("graph.hybrid_retrieve")
    @patch("graph.classify_support")
    def test_verify_statement_flow(
        self,
        mock_classify,
        mock_hybrid_retrieve,
        mock_expand,
        sample_chunks,
        sample_verdict,
    ):
        mock_expand.return_value = ["expanded 1", "expanded 2"]
        mock_hybrid_retrieve.return_value = (sample_chunks, [0.8, 0.6, 0.4])
        mock_classify.return_value = sample_verdict

        result = verify_statement("Mojżesz wyprowadził Izraelitów z Egiptu")
        assert result == sample_verdict
        assert result.verdict == Verdict.SUPPORTED
