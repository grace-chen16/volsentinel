import numpy as np
import pandas as pd
import pytest

from volsentinel import __main__ as cli
from volsentinel.agents.orchestrator import run_forecast
from volsentinel.data.live_store import LiveStore


def _rows(dates, btc, funding=None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    df = pd.DataFrame({"BTC": btc, "USDT": 1.0, "DAI": 1.0}, index=idx)
    df["BTC_funding_mean"] = funding if funding is not None else np.nan
    df["funding_src"] = "okx"
    return df


def test_upsert_new_values_win_but_gaps_never_erase(tmp_path):
    s = LiveStore(tmp_path / "live.db")
    assert s.upsert(_rows(["2026-01-01", "2026-01-02"], [100.0, 101.0], [1e-5, 2e-5])) == {"new": 2, "updated": 0}
    # a later fetch revises a price, has no funding for that day, and adds a new day
    r = s.upsert(_rows(["2026-01-02", "2026-01-03"], [102.0, 103.0]))
    assert r == {"new": 1, "updated": 1}
    df = s.load()
    assert list(df.BTC) == [100.0, 102.0, 103.0]
    assert df.at[pd.Timestamp("2026-01-02"), "BTC_funding_mean"] == 2e-5     # kept, not erased
    assert s.last_date() == pd.Timestamp("2026-01-03")
    assert s.summary()["days"] == 3 and s.summary()["funding_days"] == 2


@pytest.fixture
def live_ctx(fresh_ctx, tmp_path, monkeypatch):
    """Context with an empty live store and a fake live feed that continues the archive."""
    fresh_ctx.live_store = LiveStore(tmp_path / "live.db")
    fresh_ctx._live_cache = None
    db_last = fresh_ctx.raw.dropna(subset=["BTC"]).index.max()
    calls = []

    def fake_tail(after, today=None):
        calls.append(after)
        days = pd.date_range(after + pd.Timedelta(days=1), db_last + pd.Timedelta(days=40))
        rng = np.random.default_rng(len(calls))
        btc = fresh_ctx.raw.BTC.dropna().iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
        return _rows(days, btc, rng.normal(1e-5, 5e-6, len(days))), {"BTC": "ok"}

    monkeypatch.setattr("volsentinel.data.live.live_tail", fake_tail)
    yield fresh_ctx, db_last, calls
    fresh_ctx._live_cache = None


def test_live_fetch_is_persisted_and_extends_replay(live_ctx):
    ctx, db_last, calls = live_ctx
    res = ctx.ingest()
    assert res["counts"]["new"] == 40 and calls[0] == db_last
    stored = ctx.stored_market()
    assert stored.index.max() == db_last + pd.Timedelta(days=40)
    # a second ingest starts from the last stored day minus the overlap, not from the archive end
    ctx.ingest()
    assert calls[1] == db_last + pd.Timedelta(days=40 - ctx.settings.live_overlap_days)
    # replay now accepts a date that only the live store covers
    a = run_forecast(ctx, "q", str((db_last + pd.Timedelta(days=30)).date()))
    assert a["asof"] == str((db_last + pd.Timedelta(days=30)).date()) and a["status"] != "failed"
    # a successful live run records the store summary in the trace, not as a data-quality issue
    ctx._live_cache = None
    live = run_forecast(ctx, "q", mode="live")
    assert not any(i["feed"] == "live:stored" for i in live["data_quality"]["issues"])
    assert any("live store:" in e["text"] for e in live["trace"])


def test_live_failure_falls_back_to_stored_days(live_ctx, monkeypatch):
    ctx, db_last, _ = live_ctx
    ctx.ingest()

    def broken(after, today=None):
        raise RuntimeError("feed down")

    monkeypatch.setattr("volsentinel.data.live.live_tail", broken)
    a = run_forecast(ctx, "q", mode="live")
    assert a["asof"] > str(db_last.date())                       # used stored days, not just the archive
    assert any(i["feed"] == "live" for i in a["data_quality"]["issues"])


def test_scoring_uses_stored_live_days(live_ctx):
    ctx, db_last, _ = live_ctx
    ctx.ingest()
    a = run_forecast(ctx, "q", str((db_last + pd.Timedelta(days=10)).date()), save=False)
    a["status"] = "approved"
    ctx.memory.save(a)
    assert ctx.memory.score_due(ctx.mkt) == 0                   # archive alone cannot score it
    assert ctx.memory.score_due(ctx.stored_market()) == 1        # stored live days can


def test_cli_scheduled_flag_holds_flagged_briefs(fresh_ctx, monkeypatch, capsys):
    seen = {}

    def fake_run(ctx, *args, **kw):
        seen.update(kw)
        return {"status": "failed", "headline": "h", "explanation": "e",
                "human_review": {"required": True, "release": "hold for human review"}}

    monkeypatch.setattr("volsentinel.agents.orchestrator.run_forecast", fake_run)
    monkeypatch.setattr("volsentinel.context.Context", lambda *a, **k: fresh_ctx)
    assert cli.main(["ask", "--scheduled", "--no-news"]) == 2
    assert seen["trigger"] == "scheduled"
    assert "HELD FOR REVIEW" in capsys.readouterr().out
    # with --json the notice goes to stderr so stdout stays valid JSON
    assert cli.main(["ask", "--scheduled", "--no-news", "--json"]) == 2
    out = capsys.readouterr()
    import json
    json.loads(out.out)
    assert "HELD FOR REVIEW" in out.err
    assert cli.main(["ask", "--no-news"]) == 2 and seen["trigger"] == "interactive"
