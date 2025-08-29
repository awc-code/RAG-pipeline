from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_rag_success():
    payload = {"query": "What is Beazley Breach Response?"}
    response = client.post("/rag", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_rag_missing_query():
    payload = {"documents": ["beazley-bbr-brochure-factsheet-us.pdf"]}
    response = client.post("/rag", json=payload)
    assert response.status_code == 422


def test_rag_empty_query():
    payload = {"query": "   "}
    response = client.post("/rag", json=payload)
    assert response.status_code == 422
    assert "Query cannot be empty" in response.json()["detail"]


def test_rag_internal_error(monkeypatch):
    def mock_ask(query):
        raise RuntimeError("Pipeline failure")

    monkeypatch.setattr("main.rag_pipeline.ask", mock_ask)

    payload = {"query": "test"}
    response = client.post("/rag", json=payload)
    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]
