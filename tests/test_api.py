import pytest


@pytest.fixture(scope="module")
def client(ctx):
    from volsentinel import api
    api._ctx = ctx
    return api.app.test_client()


def test_health_and_dates(client):
    h = client.get("/api/health").get_json()
    assert h["status"] == "ok" and h["event_cards"] == 22
    d = client.get("/api/dates").get_json()
    assert d["min"] < d["max"] and len(d["episodes"]) == 22


def test_forecast_roundtrip(client):
    r = client.post("/api/forecast", json={"as_of": "2022-11-07", "headlines": ["FTX halts withdrawals"]})
    assert r.status_code == 200, r.get_data(as_text=True)
    a = r.get_json()
    assert a["forecast"]["most_likely"] in ("Calm", "Elevated", "Stressed")
    saved = client.get(f"/api/runs/{a['run_id']}").get_json()
    assert saved["answer"]["run_id"] == a["run_id"] and saved["state"]["tree"]
    assert any(x["run_id"] == a["run_id"] for x in client.get("/api/runs").get_json())


def test_forecast_validation(client):
    assert client.post("/api/forecast", json={"as_of": "1990-01-01"}).status_code == 400
    assert client.post("/api/forecast", json={"simulate_failures": ["har_forecast"]}).status_code == 400
    assert client.post("/api/forecast", json={"mode": "trade"}).status_code == 400
    assert client.get("/api/runs/..%2Fsecret").status_code == 404
    assert client.get("/").status_code == 200
