"""Fit the warning gate on validation weeks only (Checkpoint 4.1: "thresholds are
starting values, tuned on validation weeks only").

    python -m volsentinel tune --start 2021-01-04 --end 2022-12-26

1. Replay every validation week once with the gate switched off, recording each
   week's pre-gate warning and the gate's inputs (branch score, evidence support,
   HAR P(Stressed), same-family precedent, number of risk headlines).
2. Grid-search the four gate thresholds on that table. A warning or severe level
   that fails any threshold is lowered to 'watch' (still flagged, but not an alarm).
3. Objective: cost = false alarms + MISS_COST x stressed weeks without a warning.
   A missed stressed week is weighted as five false alarms, following the safety
   plan's "caution versus sensitivity" trade-off. Ties go to the simpler gate.
4. Freeze the thresholds, then look once at the held-out test weeks. The gate is
   adopted only if it lowers the cost there without losing a correct warning;
   otherwise tuned.json records the fit with "adopted": false and the defaults
   stay in force. Never re-tune after looking at the held-out result.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pandas as pd

from .config import PKG_DIR
from .evaluate import evaluate
from .memory import WARN_RANK

MISS_COST = 5.0
GRID = {
    "warn_min_score": [0.0, 0.50, 0.55, 0.60, 0.65, 0.70],
    "warn_min_support": [0.0, 0.34, 0.50, 0.67],
    "warn_min_p_stress": [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 1.01],   # 1.01 = precedent required
    "warn_min_risk_headlines": [0, 2, 3, 4, 6],
}
DEFAULTS = {"warn_min_score": 0.0, "warn_min_support": 0.0, "warn_min_p_stress": 0.0, "warn_min_risk_headlines": 0}


def apply_gate(df: pd.DataFrame, p: dict, prefix: str = "tot") -> pd.Series:
    """Gated warning level per week, reproducing ScenarioAnalyst.gate on recorded inputs."""
    raw = df[f"{prefix}_warning_raw"]
    ok = ((df[f"{prefix}_g_score"].fillna(0) >= p["warn_min_score"])
          & (df[f"{prefix}_g_support"].fillna(0) >= p["warn_min_support"])
          & (df[f"{prefix}_g_precedent"].fillna(False).astype(bool)
             | (df[f"{prefix}_g_p_stress"].fillna(0) >= p["warn_min_p_stress"]))
          & (df[f"{prefix}_g_risk"].isna() | (df[f"{prefix}_g_risk"] >= p["warn_min_risk_headlines"])))
    lowered = (raw.map(WARN_RANK) >= 2) & ~ok
    return raw.where(~lowered, "watch")


def score(df: pd.DataFrame, warning: pd.Series) -> dict:
    warned = warning.map(WARN_RANK) >= 2
    stressed = df.outcome == "Stressed"
    fa, hits = int((warned & ~stressed).sum()), int((warned & stressed).sum())
    misses = int((~warned & stressed).sum())
    return {"warnings": int(warned.sum()), "false_alarms": fa, "hits": hits, "missed_stressed": misses,
            "stressed_weeks": int(stressed.sum()),
            "false_alarm_rate": fa / warned.sum() if warned.sum() else None,
            "recall": hits / stressed.sum() if stressed.sum() else None,
            "cost": fa + MISS_COST * misses}


def search(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    keys = list(GRID)
    results = []
    for vals in itertools.product(*(GRID[k] for k in keys)):
        p = dict(zip(keys, vals))
        s = score(df, apply_gate(df, p))
        s["active_thresholds"] = sum(p[k] != DEFAULTS[k] for k in keys)
        results.append({"params": p, **s})
    results.sort(key=lambda r: (r["cost"], r["false_alarms"], r["active_thresholds"],
                                tuple(r["params"][k] for k in keys)))
    return results[0]["params"], results


def _raw_frame(ctx, start, end):
    """Replay weeks with the gate switched off, recording raw warnings and gate inputs."""
    s = ctx.settings
    saved = s.warn_gate_enabled
    s.warn_gate_enabled = False
    try:
        _, df = evaluate(ctx, start, end, return_frame=True)
    finally:
        s.warn_gate_enabled = saved
    return df


def held_out_check(df: pd.DataFrame, params: dict) -> dict:
    """Adopt the gate only if, on held-out weeks, it lowers the cost without losing a correct warning
    (the same rule Checkpoint 3.1 applies to blending precedents into the probabilities)."""
    before, after = score(df, df.tot_warning_raw), score(df, apply_gate(df, params))
    adopt = after["cost"] < before["cost"] and after["hits"] >= before["hits"]
    return {"weeks": int(len(df)), "before": before, "after": after, "adopted": bool(adopt),
            "reason": ("lower cost on held-out weeks with no lost hits" if adopt else
                       "no improvement on held-out weeks (cost not lower, or a correct warning was lost)")}


def tune(ctx, start="2021-01-04", end="2022-12-26", out_dir: Path = Path("results"),
         test_start: str | None = "2023-01-02", test_end: str | None = None) -> dict:
    s = ctx.settings
    df = _raw_frame(ctx, start, end)
    best, results = search(df)
    baseline = score(df, df.tot_warning_raw)
    chosen = score(df, apply_gate(df, best))
    cot = {"before": score(df, df.cot_warning_raw), "after": score(df, apply_gate(df, best, "cot"))}
    payload = {
        "params": best,
        "fitted_on": f"validation weeks {start} to {end}",
        "fitted_at": time.strftime("%Y-%m-%d %H:%M"),
        "objective": f"false alarms + {MISS_COST:g} x missed stressed weeks; ties -> fewer false alarms, simpler gate",
        "validation": {"weeks": int(len(df)), "before": baseline, "after": chosen, "single_chain": cot},
        "grid_size": len(results),
        "top5": [{"params": r["params"], "cost": r["cost"], "false_alarms": r["false_alarms"], "hits": r["hits"]}
                 for r in results[:5]],
        "adopted": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tuning_validation_2021_2022.csv", index=False)
    if test_start:
        # one look at the held-out period, with thresholds already frozen
        test = _raw_frame(ctx, test_start, test_end)
        test.to_csv(out_dir / "tuning_heldout_check.csv", index=False)
        payload["held_out"] = {"period": f"{test_start} to {test_end or 'latest scored week'}",
                               **held_out_check(test, best)}
        payload["adopted"] = payload["held_out"]["adopted"]
    text = json.dumps(payload, indent=1, default=float)
    (PKG_DIR / "tuned.json").write_text(text, encoding="utf-8")
    (out_dir / "tuning_validation_2021_2022.json").write_text(text, encoding="utf-8")
    if payload["adopted"]:
        for k, v in best.items():
            setattr(s, k, v)
        s.tuned_source = f"tuned.json ({payload['fitted_on']})"
    else:
        s.tuned_source = "defaults (tuned.json fitted but not adopted)"
    return payload
