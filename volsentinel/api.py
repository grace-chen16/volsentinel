"""HTTP API and web UI (Flask, matching the existing crypto-vol-api backend).

    python -m volsentinel serve --port 8000

Endpoints
  GET  /                    web UI
  GET  /api/health          data sources, LLM status
  GET  /api/dates           valid replay range and past episodes
  POST /api/forecast        run the five agents (JSON body below)
  GET  /api/runs            recent runs
  GET  /api/runs/<id>       a saved run: answer + shared state (evidence, tree, trace)
  GET  /api/metrics         evaluation metrics over scored forecasts, drift check

All endpoints are read-only with respect to markets: VolSentinel never trades.
"""
from __future__ import annotations

import logging
import threading

from flask import Flask, abort, jsonify, request, send_from_directory

from .agents.orchestrator import RunError, run_forecast
from .config import PKG_DIR
from .context import Context

log = logging.getLogger(__name__)
SIMULATABLE = {"news_search:primary", "news_search:backup", "funding_rates", "vix_fear", "stablecoin_peg",
               "retrieve_precedents", "memory_search", "market_snapshot"}

app = Flask(__name__, static_folder=None)
_ctx: Context | None = None
_lock = threading.Lock()


def ctx() -> Context:
    global _ctx
    with _lock:
        if _ctx is None:
            _ctx = Context()
    return _ctx


def bad_request(msg):
    return jsonify({"detail": msg}), 400


@app.get("/")
def index():
    return send_from_directory(PKG_DIR / "web", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", **ctx().describe()})


@app.get("/api/dates")
def dates():
    c = ctx()
    ok = c.mkt.dropna(subset=["rv7", "thr_calm", "rv30"])
    return jsonify({"min": str(ok.index.min().date()), "max": str(c.default_asof.date()),
                    "episodes": [{"id": r.id, "title": r.title, "date": str(r.date.date())}
                                 for r in c.cards.cards.itertuples()]})


@app.post("/api/forecast")
def forecast():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question") or "Is a volatility spike likely this week?")
    if len(question) > 500:
        return bad_request("question longer than 500 characters")
    mode = body.get("mode", "replay")
    if mode not in ("replay", "live"):
        return bad_request("mode must be 'replay' or 'live'")
    trigger = body.get("trigger", "interactive")
    if trigger not in ("interactive", "scheduled"):
        return bad_request("trigger must be 'interactive' or 'scheduled'")
    headlines = body.get("headlines") or []
    if not isinstance(headlines, list) or len(headlines) > 50:
        return bad_request("headlines must be a list of at most 50 strings")
    fails = body.get("simulate_failures") or []
    bad = set(fails) - SIMULATABLE
    if bad:
        return bad_request(f"cannot simulate failure of {sorted(bad)}; allowed: {sorted(SIMULATABLE)}")
    try:
        ans = run_forecast(ctx(), question, body.get("as_of"), mode, [str(h) for h in headlines], fails,
                           use_llm=bool(body.get("use_llm", True)), trigger=trigger)
    except RunError as e:
        return bad_request(str(e))
    return jsonify(ans)


@app.get("/api/runs")
def runs():
    limit = request.args.get("limit", 30, type=int)
    return jsonify(ctx().store.list(min(max(limit, 1), 200)))


@app.get("/api/runs/<run_id>")
def run(run_id):
    r = ctx().store.load(run_id)
    if r is None:
        abort(404)
    return jsonify(r)


@app.get("/api/metrics")
def metrics():
    c = ctx()
    c.memory.score_due(c.mkt)
    return jsonify({"metrics": c.memory.metrics(), "drift": c.memory.drift(c.settings.drift_window_weeks)})


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"detail": "not found"}), 404
