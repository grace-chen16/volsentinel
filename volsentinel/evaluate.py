"""Walk-forward evaluation on replayed weeks (Checkpoints 4.1 and 6.1).

For each weekly as-of date the full multi-agent pipeline runs exactly as it would
have on that day (no look-ahead: HAR is refit on past data, precedents must be
30+ days old, memory only sees forecasts already scored). The run's forecast is
then scored against the realized next-week regime.

Reports
  correctness   Brier vs naive and HAR-only, quiet vs stressed weeks
  calibration   stated top-class probability vs observed frequency
  safety        missed-stress rate, false-alarm rate of warnings (ToT vs single chain)
  groundedness  share of numeric claims verified against a source
  reliability   fallback rate, human-review rate
  efficiency    p95 latency, tool calls, node evaluations
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agents.orchestrator import run_forecast
from .data.market import REGIMES, brier
from .memory import WARN_RANK, ExperienceMemory
from .scenarios import FAMILY
from .state import StateStore


def direction_match(family, warning, har_modal, outcome) -> bool:
    """Did the scenario's implied direction relative to HAR match what happened?"""
    m, y = REGIMES.index(har_modal), REGIMES.index(outcome)
    up = family is not None and FAMILY[family].direction != "calm" and WARN_RANK[warning] >= 1
    return (y > m) if up else (y <= m)


def _gate_cols(prefix, gate, level) -> dict:
    """Pre-gate warning and the gate's inputs, so the gate can be refitted without rerunning."""
    f = (gate or {}).get("features", {})
    return {f"{prefix}_warning_raw": (gate or {}).get("raw", level),
            f"{prefix}_g_score": f.get("score"), f"{prefix}_g_support": f.get("support"),
            f"{prefix}_g_p_stress": f.get("p_stress"), f"{prefix}_g_precedent": f.get("same_family_precedent"),
            f"{prefix}_g_risk": f.get("risk_headlines")}


def evaluate(ctx, start="2023-01-02", end=None, step_days=7, out: Path | None = None, limit=None,
             return_frame: bool = False):
    end = pd.Timestamp(end) if end else ctx.mkt[ctx.mkt.fwd_regime >= 0].index.max()
    dates = pd.date_range(pd.Timestamp(start), end, freq=f"{step_days}D")
    if limit:
        dates = dates[:limit]
    # fresh experience memory so the replay learns forward in time only
    mem_path = ctx.settings.runs_dir / f"eval_memory_{int(time.time())}.db"
    live_memory, ctx.memory = ctx.memory, ExperienceMemory(mem_path)
    live_store, ctx.store = ctx.store, StateStore(ctx.settings.runs_dir / "eval")
    rows = []
    try:
        for i, d in enumerate(dates):
            if d not in ctx.mkt.index or ctx.mkt.at[d, "fwd_regime"] < 0:
                continue
            a = run_forecast(ctx, "Is a volatility spike likely this week?", str(d.date()), "replay",
                             use_llm=False, save=True, baseline=True)
            ctx.memory.score_due(ctx.mkt.loc[:d - pd.Timedelta(days=7)])   # only outcomes known by this date
            y = int(ctx.mkt.at[d, "fwd_regime"])
            p = [a["forecast"]["probabilities"][r] for r in REGIMES]
            b = a["baseline_single_chain"]
            rows.append({
                "asof": str(d.date()), "outcome": REGIMES[y], "har_modal": a["forecast"]["most_likely"],
                "brier_har": brier(p, y), "brier_naive": brier(a["forecast"]["naive_probabilities"], y),
                "p_stress": p[2], "top_p": max(p), "hit": a["forecast"]["most_likely"] == REGIMES[y],
                "tot_family": a["scenario"]["family"], "tot_warning": a["scenario"]["warning_level"],
                "tot_dir": direction_match(a["scenario"]["family"], a["scenario"]["warning_level"],
                                           a["forecast"]["most_likely"], REGIMES[y]),
                "cot_family": b["family"], "cot_warning": b["warning_level"], "cot_valid": bool(b["citation_ok"]),
                "cot_dir": direction_match(b["family"], b["warning_level"], a["forecast"]["most_likely"], REGIMES[y]),
                "status": a["status"], "review": a["human_review"]["required"],
                "critic_rejections": a["critic"]["rejections"], "grounded": a["metrics"]["grounded_share"],
                "latency": a["metrics"]["latency_s"], "tool_calls": a["metrics"]["tool_calls"],
                "node_evals": a["metrics"]["node_evals"], "n_precedents": len(a["precedents"]),
                **_gate_cols("tot", a["scenario"].get("warning_gate"), a["scenario"]["warning_level"]),
                **_gate_cols("cot", b.get("warning_gate"), b["warning_level"]),
            })
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(dates)} weeks", flush=True)
    finally:
        ctx.memory, ctx.store = live_memory, live_store
    df = pd.DataFrame(rows)
    rep = summarize(df)
    rep["period"] = {"start": str(dates[0].date()), "end": str(dates[-1].date()), "weeks": int(len(df))}
    s = ctx.settings
    rep["warning_gate"] = {"enabled": s.warn_gate_enabled, "source": s.tuned_source,
                           "params": {k: getattr(s, k) for k in ("warn_min_score", "warn_min_support",
                                                                 "warn_min_p_stress", "warn_min_risk_headlines")}}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        df.to_csv(out.with_suffix(".csv"), index=False)
    return (rep, df) if return_frame else rep


def summarize(df: pd.DataFrame) -> dict:
    stress = df[df.outcome == "Stressed"]
    quiet = df[df.outcome != "Stressed"]

    def warn_stats(col):
        warned = df[df[col].map(WARN_RANK) >= 2]
        return {"warnings_issued": int(len(warned)),
                "false_alarm_rate": float((warned.outcome != "Stressed").mean()) if len(warned) else None,
                "stressed_weeks_warned": float((stress[col].map(WARN_RANK) >= 2).mean()) if len(stress) else None}

    from .tune import score as gate_score      # local import: tune imports this module
    bands = pd.cut(df.top_p, [0, 0.5, 0.6, 0.7, 0.8, 1.0])
    return {
        "warning_gate_effect": {"tot_before": gate_score(df, df.tot_warning_raw),
                                "tot_after": gate_score(df, df.tot_warning)},
        "correctness": {"brier_har": df.brier_har.mean(), "brier_naive": df.brier_naive.mean(),
                        "brier_har_quiet": quiet.brier_har.mean(), "brier_naive_quiet": quiet.brier_naive.mean(),
                        "brier_har_stressed": stress.brier_har.mean() if len(stress) else None,
                        "brier_naive_stressed": stress.brier_naive.mean() if len(stress) else None,
                        "stressed_weeks": int(len(stress))},
        "calibration": [{"band": str(b), "n": int(len(g)), "stated": float(g.top_p.mean()), "observed": float(g.hit.mean())}
                        for b, g in df.groupby(bands, observed=True)],
        "safety": {"missed_stress_rate": float((stress.har_modal == "Calm").mean()) if len(stress) else None,
                   "tot": warn_stats("tot_warning"), "single_chain": warn_stats("cot_warning")},
        "tot_vs_single_chain": {"direction_match_tot": float(df.tot_dir.mean()),
                                "direction_match_single_chain": float(df.cot_dir.mean()),
                                "single_chain_citation_failures": int((~df.cot_valid).sum()),
                                "tot_node_evals_mean": float(df.node_evals.mean())},
        "groundedness": {"grounded_share_mean": float(df.grounded.dropna().mean()),
                         "critic_rejection_rate": float((df.critic_rejections > 0).mean())},
        "reliability": {"status_counts": df.status.value_counts().to_dict(),
                        "human_review_rate": float(df.review.mean())},
        "efficiency": {"latency_p95_s": float(np.percentile(df.latency, 95)), "tool_calls_mean": float(df.tool_calls.mean()),
                       "tool_calls_max": int(df.tool_calls.max())},
        "retrieval": {"weeks_with_precedent": float((df.n_precedents > 0).mean())},
    }
