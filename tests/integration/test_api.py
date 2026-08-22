from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestAPIEndpoints:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_serve_index(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_favicon_route(self, client: TestClient):
        response = client.get("/favicon.ico")
        assert response.status_code in [200, 304]

    def test_get_chapter_valid(self, client: TestClient):
        response = client.get("/chapter", params={"book": "Wj", "chapter": 3})
        assert response.status_code == 200
        data = response.json()
        assert data["book_abbr"] == "Wj"
        assert data["chapter"] == 3
        assert len(data["verses"]) > 0
        assert data["verses"][0]["verse"] == 1
        assert data["prev_chapter"]["chapter"] == 2
        assert data["next_chapter"]["chapter"] == 4

    def test_get_chapter_not_found(self, client: TestClient):
        response = client.get("/chapter", params={"book": "NieznanaKsiega", "chapter": 1})
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

        response_invalid_num = client.get("/chapter", params={"book": "Rdz", "chapter": 999})
        assert response_invalid_num.status_code == 404

    def test_get_chapter_validation_error(self, client: TestClient):
        response = client.get("/chapter", params={"book": "Wj", "chapter": 0})
        assert response.status_code == 422

        response_empty_book = client.get("/chapter", params={"book": "   ", "chapter": 1})
        assert response_empty_book.status_code == 400

    @patch("main.verify_statement")
    def test_verify_endpoint_success(self, mock_verify, client: TestClient, sample_verdict):
        mock_verify.return_value = sample_verdict

        payload = {"statement": "Mojżesz wyprowadził Izraelitów z Egiptu."}
        response = client.post("/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "directly_supported"
        assert data["confidence"] == 0.98
        assert len(data["citations"]) == 1
        assert data["citations"][0]["ref"] == "Wj 3:10"

    def test_verify_empty_statement(self, client: TestClient):
        response = client.post("/verify", json={"statement": "   "})
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"].lower()

    def test_verify_statement_too_long(self, client: TestClient):
        long_statement = "A" * 501
        response = client.post("/verify", json={"statement": long_statement})
        assert response.status_code == 400
        assert "too long" in response.json()["detail"].lower()

    def test_verify_missing_payload(self, client: TestClient):
        response = client.post("/verify", json={})
        assert response.status_code == 422

    @patch("main.verify_statement")
    def test_verify_pipeline_error_handling(self, mock_verify, client: TestClient):
        mock_verify.side_effect = RuntimeError("Qdrant connection failure")
        response = client.post("/verify", json={"statement": "Test error statement"})
        assert response.status_code == 502
        assert "verification failed" in response.json()["detail"].lower()
