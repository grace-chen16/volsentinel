"""MCP server: the shared state store and the forecaster, exposed read-only.

    python -m volsentinel mcp        (stdio transport)

Other agents or MCP clients can request a forecast and read any run's evidence,
scenario tree and reasoning trace. Nothing here can trade or change market state.
"""
from __future__ import annotations

import threading

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("volsentinel")
_ctx = None
_lock = threading.Lock()


def ctx():
    global _ctx
    with _lock:
        if _ctx is None:
            from .context import Context
            _ctx = Context()
    return _ctx


@mcp.tool()
def forecast(question: str = "Is a volatility spike likely this week?", as_of: str | None = None,
             mode: str = "replay") -> dict:
    """Run VolSentinel for an as-of date (replay, YYYY-MM-DD) or today (mode='live').
    Returns regime probabilities, the leading scenario, the cited explanation and review flags."""
    from .agents.orchestrator import run_forecast
    a = run_forecast(ctx(), question, as_of, "live" if mode == "live" else "replay")
    keep = ("run_id", "asof", "status", "headline", "forecast", "scenario", "alternative", "explanation",
            "precedents", "human_review", "confidence", "metrics", "disclaimer")
    return {k: a.get(k) for k in keep}


@mcp.tool()
def list_runs(limit: int = 20) -> list:
    """Most recent runs with their regime, warning level and review status."""
    return ctx().store.list(min(max(limit, 1), 200))


@mcp.tool()
def get_evidence(run_id: str) -> dict:
    """Every number a run's tools produced (E1, E2, ...), with its source tool and agent."""
    r = ctx().store.load(run_id)
    return r["state"]["evidence"] if r else {"error": "run not found"}


@mcp.tool()
def get_tree(run_id: str) -> dict:
    """The run's Tree-of-Thought nodes with scores, rubric components and status."""
    r = ctx().store.load(run_id)
    return r["state"]["tree"] if r else {"error": "run not found"}


@mcp.tool()
def get_trace(run_id: str) -> list:
    """The run's reasoning trace: each agent's reason / act / observe step and hand-off."""
    r = ctx().store.load(run_id)
    return r["state"]["log"] if r else [{"error": "run not found"}]


@mcp.tool()
def evaluation_metrics() -> dict:
    """Brier score, calibration, missed-stress and false-alarm rates over scored forecasts."""
    c = ctx()
    c.memory.score_due(c.mkt)
    return {"metrics": c.memory.metrics(), "drift": c.memory.drift(c.settings.drift_window_weeks)}
