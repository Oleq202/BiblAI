import pytest

from backend.eval.generate_dashboard import (
    compute_metrics,
    generate_dashboard_html,
    load_jsonl,
)


@pytest.mark.unit
class TestEvalDashboard:
    def test_load_jsonl_nonexistent_file(self, tmp_path):
        res = load_jsonl(tmp_path / "does_not_exist.jsonl")
        assert res == []

    def test_load_jsonl_valid_file(self, tmp_path):
        file = tmp_path / "test.jsonl"
        file.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        res = load_jsonl(file)
        assert len(res) == 2
        assert res[0]["a"] == 1
        assert res[1]["b"] == 2

    def test_compute_metrics_empty(self):
        metrics = compute_metrics([], [], [])
        assert metrics["total_claims"] == 0
        assert metrics["accuracy_pct"] == 0.0
        assert metrics["categories_count"] == 0
        assert metrics["retrieval"]["bi_encoder_recall"] == 0.0
        assert metrics["judge"]["verdict_correct_pct"] == 0.0

    def test_compute_metrics_with_data(self):
        accuracy_data = [
            {
                "id": "t1",
                "statement": "Bóg stworzył świat",
                "category": "supported_explicit",
                "expected_verdict": "directly_supported",
                "actual_verdict": "directly_supported",
                "correct": True,
                "confidence": 1.0,
                "citations": [{"ref": "Rdz 1:1", "quote": "Na początku", "relation": "supports"}],
                "reasoning": "Rdz 1:1 potwierdza.",
            },
            {
                "id": "t2",
                "statement": "Jezus urodził się w Rzymie",
                "category": "contradicted",
                "expected_verdict": "directly_contradicted",
                "actual_verdict": "directly_contradicted",
                "correct": True,
                "confidence": 1.0,
                "citations": [],
                "reasoning": "Urodził się w Betlejem.",
            },
            {
                "id": "t3",
                "statement": "Internet w Biblii",
                "category": "unrelated",
                "expected_verdict": "not_directly_stated",
                "actual_verdict": "directly_supported",
                "correct": False,
                "confidence": 0.5,
                "citations": [],
                "reasoning": "Brak wzmianki.",
            },
        ]

        retrieval_data = [
            {"id": "t1", "bi_encoder_top10_hit": True, "rerank_top5_hit": True},
            {"id": "t2", "bi_encoder_top10_hit": True, "rerank_top5_hit": False},
        ]

        judge_data = [
            {
                "id": "t1",
                "judge": {
                    "verdict_is_correct": True,
                    "reasoning_follows_from_citations": True,
                    "issues": "None",
                },
            },
            {
                "id": "t2",
                "judge": {
                    "verdict_is_correct": True,
                    "reasoning_follows_from_citations": False,
                    "issues": "Reasoning introduces external facts",
                },
            },
        ]

        metrics = compute_metrics(accuracy_data, retrieval_data, judge_data)

        assert metrics["total_claims"] == 3
        assert metrics["correct_count"] == 2
        assert metrics["accuracy_pct"] == 66.7
        assert metrics["categories_count"] == 3

        # Confusion matrix checks
        cm = metrics["confusion_matrix"]
        assert cm["directly_supported"]["directly_supported"] == 1
        assert cm["directly_contradicted"]["directly_contradicted"] == 1
        assert cm["not_directly_stated"]["directly_supported"] == 1

        # Retrieval checks
        assert metrics["retrieval"]["bi_encoder_hits"] == 2
        assert metrics["retrieval"]["rerank_hits"] == 1
        assert metrics["retrieval"]["bi_encoder_recall"] == 100.0
        assert metrics["retrieval"]["rerank_recall"] == 50.0

        # Judge checks
        assert metrics["judge"]["verdict_correct_pct"] == 100.0
        assert metrics["judge"]["reasoning_faithful_pct"] == 50.0
        assert metrics["judge"]["issues_count"] == 1

    def test_generate_dashboard_html(self):
        metrics = compute_metrics([], [], [])
        html_str = generate_dashboard_html(metrics)
        assert "<!DOCTYPE html>" in html_str
        assert "BiblAI - Raport Ewaluacji" in html_str
        assert "Macierz Pomyłek" in html_str
        assert "Eksplorator Zbadanych Przypadków Testowych" in html_str

    def test_generate_dashboard_file(self, tmp_path):
        from backend.eval.generate_dashboard import generate_dashboard
        out = tmp_path / "dashboard.html"
        generate_dashboard(output_path=out)
        assert out.exists()
        assert "<!DOCTYPE html>" in out.read_text(encoding="utf-8")
