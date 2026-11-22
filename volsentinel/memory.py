"""Long-term experience memory (Checkpoint 2): each past forecast, a summary of its
reasoning, and the realized outcome once its seven-day window has closed.

Only verified records are written (critic-approved answers or grounded HAR-only
fallbacks). Recall respects a time guard: a past forecast is visible to a run
only if its outcome window closed before that run's as-of date.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data.market import FP_COLS, FP_SCALE, REGIMES, brier

SCHEMA = """
create table if not exists forecasts (
  run_id text primary key, asof text, created real, mode text,
  p_calm real, p_elev real, p_stress real, p_naive text,
  most_likely text, scenario text, warning text, tot_score real,
  summary text, fingerprint text, status text,
  critic_rejections integer, human_review integer, fallback integer, latency_s real, tool_calls integer,
  grounded_share real,
  outcome_regime text, outcome_rv7 real, brier real, brier_naive real, scored_at real
);
"""
WARN_RANK = {"none": 0, "watch": 1, "warning": 2, "severe": 3}


class ExperienceMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._con() as c:
            c.executescript(SCHEMA)

    def _con(self):
        return sqlite3.connect(self.path, timeout=10)

    # ---- write --------------------------------------------------------------
    def save(self, ans: dict) -> bool:
        if ans["status"] not in ("approved", "approved_low_confidence", "no_dominant_scenario", "har_only_fallback"):
            return False
        f = ans["forecast"]
        p = f["probabilities"]
        row = (ans["run_id"], ans["asof"], time.time(), ans["mode"], p["Calm"], p["Elevated"], p["Stressed"],
               json.dumps(f.get("naive_probabilities")), f["most_likely"], ans["scenario"]["name"],
               ans["scenario"]["warning_level"], ans["scenario"].get("score"), ans["explanation"][:1500],
               json.dumps(ans.get("fingerprint", {})), ans["status"], ans["critic"]["rejections"],
               int(ans["human_review"]["required"]), int(ans["status"] == "har_only_fallback"),
               ans["metrics"]["latency_s"], ans["metrics"]["tool_calls"], ans["metrics"].get("grounded_share"))
        with self._lock, self._con() as c:
            c.execute("insert or replace into forecasts (run_id, asof, created, mode, p_calm, p_elev, p_stress, p_naive, "
                      "most_likely, scenario, warning, tot_score, summary, fingerprint, status, critic_rejections, "
                      "human_review, fallback, latency_s, tool_calls, grounded_share) "
                      "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        return True

    def score_due(self, mkt: pd.DataFrame) -> int:
        """Score every forecast whose seven-day window has closed in the market data."""
        n = 0
        with self._lock, self._con() as c:
            rows = c.execute("select run_id, asof, p_calm, p_elev, p_stress, p_naive from forecasts "
                             "where outcome_regime is null").fetchall()
            for run_id, asof, pc, pe, ps, pn in rows:
                d = pd.Timestamp(asof)
                if d not in mkt.index or int(mkt.at[d, "fwd_regime"]) < 0:
                    continue
                y = int(mkt.at[d, "fwd_regime"])
                bn = brier(json.loads(pn), y) if pn and pn != "null" else None
                c.execute("update forecasts set outcome_regime=?, outcome_rv7=?, brier=?, brier_naive=?, scored_at=? "
                          "where run_id=?", (REGIMES[y], float(mkt.at[d, "fwd_rv7"]), brier([pc, pe, ps], y), bn,
                                             time.time(), run_id))
                n += 1
        return n

    # ---- read ---------------------------------------------------------------
    def frame(self) -> pd.DataFrame:
        with self._con() as c:
            return pd.read_sql("select * from forecasts order by asof", c)

    def search_similar(self, fingerprint: dict, asof: pd.Timestamp, k: int = 3) -> dict:
        """Scored past forecasts in similar market conditions, plus my own over/under-call record."""
        df = self.frame()
        if df.empty:
            return {"similar": [], "n_scored": 0, "calibration_note": "no past forecasts in memory"}
        df["asof_ts"] = pd.to_datetime(df["asof"])
        df = df[df.outcome_regime.notna() & (df.asof_ts <= asof - pd.Timedelta(days=7))]
        if df.empty:
            return {"similar": [], "n_scored": 0, "calibration_note": "no scored forecasts before this date"}
        fps = df.fingerprint.map(json.loads)
        sims = []
        for fp in fps:
            cols = [c for c in FP_COLS if fingerprint.get(c) is not None and fp.get(c) is not None]
            z = sum(((fp[c] - fingerprint[c]) / FP_SCALE[c]) ** 2 for c in cols) if cols else 50
            sims.append(float(np.exp(-z / (2 * max(1, len(cols))))))
        df["sim"] = sims
        top = df.sort_values("sim", ascending=False).head(k)
        stress_idx = REGIMES.index("Stressed")
        over = ((top.warning.map(WARN_RANK) >= 2) & (top.outcome_regime != "Stressed")).sum()
        under = ((top.warning.map(WARN_RANK) <= 1) & (top.outcome_regime == "Stressed")).sum()
        note = (f"In {len(top)} similar past setups I issued {int(over)} warning(s) that did not materialise "
                f"and missed {int(under)} stressed week(s).")
        return {"n_scored": int(len(df)), "calibration_note": note, "stress_index": stress_idx,
                "similar": [{"run_id": r.run_id, "asof": r.asof, "similarity": round(r.sim, 2),
                             "forecast": r.most_likely, "warning": r.warning, "scenario": r.scenario,
                             "outcome": r.outcome_regime, "brier": round(r.brier, 3)} for r in top.itertuples()]}

    def metrics(self) -> dict:
        """Evaluation metrics from Checkpoint 6.1 over all scored forecasts."""
        df = self.frame()
        out = {"n_forecasts": int(len(df))}
        if df.empty:
            return out
        out["fallback_rate"] = float(df.fallback.mean())
        out["human_review_rate"] = float(df.human_review.mean())
        out["latency_p95_s"] = float(np.percentile(df.latency_s.dropna(), 95)) if df.latency_s.notna().any() else None
        out["critic_rejection_rate"] = float((df.critic_rejections > 0).mean())
        out["grounded_share_mean"] = float(df.grounded_share.dropna().mean()) if df.grounded_share.notna().any() else None
        s = df[df.outcome_regime.notna()]
        out["n_scored"] = int(len(s))
        if s.empty:
            return out
        out["brier"] = float(s.brier.mean())
        out["brier_naive"] = float(s.brier_naive.dropna().mean()) if s.brier_naive.notna().any() else None
        quiet, stress = s[s.outcome_regime != "Stressed"], s[s.outcome_regime == "Stressed"]
        out["brier_quiet"] = float(quiet.brier.mean()) if len(quiet) else None
        out["brier_stressed"] = float(stress.brier.mean()) if len(stress) else None
        out["missed_stress_rate"] = float((stress.most_likely == "Calm").mean()) if len(stress) else None
        warned = s[s.warning.map(WARN_RANK) >= 2]
        out["false_alarm_rate"] = float((warned.outcome_regime != "Stressed").mean()) if len(warned) else None
        # calibration: stated top-class probability vs observed hit rate, by band
        top_p = s[["p_calm", "p_elev", "p_stress"]].max(axis=1)
        hit = s.apply(lambda r: r.most_likely == r.outcome_regime, axis=1)
        bands = pd.cut(top_p, [0, 0.5, 0.6, 0.7, 0.8, 1.0])
        out["calibration"] = [{"band": str(b), "n": int(len(g)), "stated": float(top_p[g.index].mean()),
                               "observed": float(hit[g.index].mean())}
                              for b, g in s.groupby(bands, observed=True) if len(g)]
        return out

    def drift(self, window: int = 8) -> dict:
        """Did calibration or the critic's rejection rate worsen over the last `window` scored weeks?"""
        s = self.frame()
        s = s[s.outcome_regime.notna()].sort_values("asof")
        if len(s) < 2 * window:
            return {"drift": False, "reason": f"only {len(s)} scored forecasts (need {2 * window})"}
        recent, prior = s.tail(window), s.iloc[-2 * window:-window]
        worse_brier = recent.brier.mean() > prior.brier.mean() * 1.25
        worse_critic = (recent.critic_rejections > 0).mean() > (prior.critic_rejections > 0).mean() + 0.25
        return {"drift": bool(worse_brier or worse_critic),
                "brier_recent": float(recent.brier.mean()), "brier_prior": float(prior.brier.mean()),
                "critic_rej_recent": float((recent.critic_rejections > 0).mean()),
                "critic_rej_prior": float((prior.critic_rejections > 0).mean())}
