import pytest

from campus_assistant.service import AssistantService
from campus_assistant.web import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import shutil
    from campus_assistant.service import ROOT
    d = tmp_path_factory.mktemp("data")
    for f in ("announcements.jsonl", "meta.json"):
        shutil.copy(ROOT / "data" / f, d / f)
    app = create_app(AssistantService(d))
    return app.test_client()


def test_endpoints(client):
    assert client.get("/").status_code == 200
    s = client.get("/api/stats").get_json()
    assert s["total"] >= 300
    assert client.get("/api/announcements?category=exam").get_json()[0]["category"] == "exam"
    r = client.post("/api/ask", json={"question": "Yarın etkinlik var mı?"}).get_json()
    assert "answer" in r and "sources" in r
    assert client.post("/api/ask", json={}).status_code == 400


def test_add_announcement(client):
    r = client.post("/api/announcements", json={"title": "Satranç turnuvası",
                                                "text": "Satranç Kulübü turnuvası 13 Kasım 2026 saat 15:00'te Kültür Merkezi'nde."})
    assert r.status_code == 201 and r.get_json()["category"] == "event"
    ans = client.post("/api/ask", json={"question": "Satranç turnuvası ne zaman?"}).get_json()
    assert "13 Kasım" in ans["answer"]
