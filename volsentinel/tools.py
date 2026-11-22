"""Read-only tool registry.

Guardrails enforced here, for every agent:
  * allow-list: only the tools below exist; none can trade or hold keys;
  * budget: at most 15 tool calls and 90 seconds per run (cached results are free);
  * errors are returned as observations the ReAct loop can reason about;
  * every number a tool returns is written to the shared state as cited evidence.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data.market import FP_COLS, HAR, REGIMES, naive_probs, regime_probs
from .data.news import categorize, select_risk_headlines
from .safety import sanitize_text
from .state import RunState


def pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.0%}"


def signed(x, nd=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:+.{nd}f}"


def num(x, nd=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _f(x):
    return None if x is None or pd.isna(x) else float(x)


class ToolError(RuntimeError):
    pass


@dataclass
class Observation:
    tool: str
    ok: bool
    summary: str
    data: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)      # key -> evidence id
    error: str | None = None
    cached: bool = False


@dataclass
class RunInputs:
    """Everything a run's tools may read."""
    ctx: Any
    mkt: pd.DataFrame
    asof: pd.Timestamp
    mode: str                                     # replay | live
    user_headlines: list[str] = field(default_factory=list)
    simulate_failures: set = field(default_factory=set)
    now: pd.Timestamp | None = None

    def row(self):
        return self.mkt.loc[self.asof]

    def fingerprint(self) -> dict:
        r = self.row()
        return {c: _f(r[c]) for c in FP_COLS}


# ----------------------------------------------------------------------------- tools
def t_har_forecast(ri: RunInputs, **_):
    r = ri.row()
    har = HAR().fit(ri.mkt, ri.asof)
    mu, sd = har.predict(ri.mkt, ri.asof)
    p = regime_probs(mu, sd, r.thr_calm, r.thr_stress)
    pn = naive_probs(ri.mkt, ri.asof, sd)
    point = float(np.exp(mu))
    band = REGIMES[int(np.argmax(p))]
    data = {"point_rv7": point, "mu": mu, "sigma": sd, "probs": p.tolist(), "naive_probs": pn.tolist(),
            "thr_calm": float(r.thr_calm), "thr_stress": float(r.thr_stress), "rv7_now": float(r.rv7),
            "regime_now": REGIMES[int(r.regime_now)], "modal": band, "n_train": har.n,
            "beta": [float(b) for b in har.beta]}
    ev = [("rv7_now", r.rv7, "Trailing 7-day realized vol (annualized)", pct(r.rv7)),
          ("thr_calm", r.thr_calm, "Calm threshold (40th pct of trailing 2y rv7)", pct(r.thr_calm)),
          ("thr_stress", r.thr_stress, "Stressed threshold (85th pct of trailing 2y rv7)", pct(r.thr_stress)),
          ("har_point", point, "HAR point forecast of next-week realized vol", pct(point)),
          ("p_calm", p[0], "HAR P(Calm)", pct(p[0])), ("p_elevated", p[1], "HAR P(Elevated)", pct(p[1])),
          ("p_stressed", p[2], "HAR P(Stressed)", pct(p[2])), ("har_modal", band, "HAR most likely regime", band)]
    s = (f"HAR point forecast {pct(point)} vs Calm<{pct(r.thr_calm)}, Stressed>{pct(r.thr_stress)}; "
         f"P = Calm {pct(p[0])} / Elevated {pct(p[1])} / Stressed {pct(p[2])} -> {band} band")
    return data, ev, s


def t_market_snapshot(ri: RunInputs, **_):
    r = ri.row()
    data = {k: _f(r[k]) for k in ["BTC", "ret7", "ret30", "rv1", "rv7", "rv30", "rv7_pct"]}
    ev = [("btc_price", r.BTC, "BTC close", f"${r.BTC:,.0f}"),
          ("ret7", r.ret7, "BTC 7-day log return", f"{r.ret7:+.1%}"),
          ("ret30", r.ret30, "BTC 30-day log return", f"{r.ret30:+.1%}"),
          ("rv1", r.rv1, "1-day realized vol (annualized |return|)", pct(r.rv1)),
          ("rv30", r.rv30, "30-day realized vol", pct(r.rv30)),
          ("rv7_pct", r.rv7_pct, "rv7 percentile in trailing 2 years", num(r.rv7_pct, 2))]
    s = (f"BTC ${r.BTC:,.0f}, 7d return {pct(r.ret7)}, 30d {pct(r.ret30)}; rv1 {pct(r.rv1)}, "
         f"rv7 percentile {num(r.rv7_pct, 2)}")
    return data, ev, s


def t_funding_rates(ri: RunInputs, **_):
    r = ri.row()
    if pd.isna(r.BTC_funding_mean):
        raise ToolError("funding rate missing for the as-of date")
    data = {"funding_mean": float(r.BTC_funding_mean), "funding_z": _f(r.funding_z)}
    ev = [("funding_mean", r.BTC_funding_mean, "BTC perpetual funding rate (daily mean)", f"{r.BTC_funding_mean * 1e4:.2f} bp"),
          ("funding_z", r.funding_z, "Funding z-score vs trailing 90 days", signed(r.funding_z))]
    tag = ("far above" if r.funding_z > 2 else "far below" if r.funding_z < -2 else "within") if pd.notna(r.funding_z) else "unknown vs"
    return data, ev, f"funding {r.BTC_funding_mean * 1e4:.2f} bp, z {signed(r.funding_z)} ({tag} its 90-day range)"


def t_vix_fear(ri: RunInputs, **_):
    r = ri.row()
    if pd.isna(r.vix) and pd.isna(r.fear):
        raise ToolError("VIX and Fear & Greed both missing")
    data = {"vix": _f(r.vix), "vix_chg5": _f(r.vix_chg5), "fear": _f(r.fear)}
    ev = [("vix", r.vix, "VIX close", num(r.vix)), ("vix_chg5", r.vix_chg5, "VIX 5-day change", pct(r.vix_chg5)),
          ("fear", r.fear, "Crypto Fear & Greed index", num(r.fear, 0))]
    return data, ev, f"VIX {num(r.vix)} ({pct(r.vix_chg5)} over 5d), Fear & Greed {num(r.fear, 0)}"


def t_stablecoin_peg(ri: RunInputs, **_):
    r = ri.row()
    if pd.isna(r.stable_min):
        raise ToolError("no stablecoin prices for the as-of window")
    coins = {c: _f(r.get(c)) for c in ("USDT", "USDC", "DAI") if c in r.index and pd.notna(r.get(c))}
    ev = [("stable_min", r.stable_min, "Lowest stablecoin close (USDT/USDC/DAI, last 3 days)", f"${r.stable_min:.4f}")]
    return {"stable_min": float(r.stable_min), "coins": coins}, ev, f"lowest stablecoin close ${r.stable_min:.4f}"


HISTORY_SIGNALS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "funding_spike": lambda d: d.funding_z > 2,
    "funding_crash": lambda d: d.funding_z < -2,
    "vix_shock": lambda d: d.vix_chg5 > 0.20,
    "vol_jump": lambda d: d.rv1 > 2.5 * d.rv7,
    "drawdown": lambda d: d.ret7 < -0.10,
}


def t_history_query(ri: RunInputs, signal: str = "funding_spike", **_):
    if signal not in HISTORY_SIGNALS:
        raise ToolError(f"unknown signal {signal!r}; allowed: {sorted(HISTORY_SIGNALS)}")
    d = ri.mkt[(ri.mkt.index <= ri.asof - pd.Timedelta(days=7)) & (ri.mkt.fwd_regime >= 0)]
    hit = d[HISTORY_SIGNALS[signal](d).fillna(False)]
    # distinct episodes: first hit after a 14-day gap
    starts = hit.index[np.r_[True, np.diff(hit.index.values).astype("timedelta64[D]").astype(int) > 14]] if len(hit) else []
    ep = d.loc[list(starts)]
    if len(ep) == 0:
        return ({"signal": signal, "episodes": 0}, [(f"hist_{signal}_n", 0, f"Past episodes of {signal}", "0")],
                f"no past {signal} episodes before this date")
    dist = np.bincount(ep.fwd_regime.astype(int), minlength=3) / len(ep)
    base = np.bincount(d.fwd_regime.astype(int), minlength=3) / len(d)
    data = {"signal": signal, "episodes": int(len(ep)), "next_week_dist": dist.tolist(), "base_rate": base.tolist(),
            "examples": [str(x.date()) for x in ep.index[-5:]]}
    ev = [(f"hist_{signal}_n", len(ep), f"Past episodes of {signal}", str(len(ep))),
          (f"hist_{signal}_stress", dist[2], f"Share of {signal} episodes followed by a Stressed week", pct(dist[2])),
          ("hist_base_stress", base[2], "Base rate of Stressed weeks", pct(base[2]))]
    s = (f"{len(ep)} past {signal} episodes: next week Stressed {pct(dist[2])} of the time "
         f"(base rate {pct(base[2])}); most recent {', '.join(data['examples'][-2:])}")
    return data, ev, s


def _headline_frame(titles, now) -> pd.DataFrame:
    return pd.DataFrame({"title": titles, "dt": [now] * len(titles), "date": [now.normalize()] * len(titles)})


def t_news_search(ri: RunInputs, feed: str = "primary", days: int = 3, **_):
    days = int(min(max(days, 1), 7))
    backup_aggregates = None
    if ri.mode == "live":
        from .data.live import rss_headlines
        h = rss_headlines(feed)
        now = ri.now or pd.Timestamp.utcnow().tz_localize(None)
        h = h[h.dt > now - pd.Timedelta(days=days)]
        source = h.sourceDomain.iloc[0] if len(h) else feed
    elif feed == "primary":
        if ri.ctx.news is None:
            if not ri.user_headlines:
                raise ToolError(f"news archive unavailable: {ri.ctx.news_error}")
            h, source = pd.DataFrame(columns=["title", "dt", "date"]), "user-supplied headlines"
        else:
            h = ri.ctx.news.window(ri.asof, days)
            source = "CryptoPanic archive"
    else:
        # backup in replay: the project's daily news-flow aggregates (counts, not text)
        r = ri.row()
        backup_aggregates = {"news_negative": _f(r.news_negative), "news_negative_z": _f(r.news_negative_z),
                             "news_volume": _f(r.news_volume), "news_volume_z": _f(r.news_volume_z)}
        if all(v is None for v in backup_aggregates.values()):
            raise ToolError("backup news aggregates missing for the as-of date")
        h = pd.DataFrame(columns=["title", "dt", "date"])
        source = "daily news-flow aggregates (project DB)"
    if ri.user_headlines and feed == "primary":
        now = ri.asof + pd.Timedelta(hours=23)
        h = pd.concat([h, _headline_frame(ri.user_headlines, now)], ignore_index=True)

    clean, flagged = [], []
    for title, dt in zip(h["title"].tolist(), h["dt"].tolist()):
        t, bad = sanitize_text(title)
        (flagged if bad else clean).append({"title": t, "dt": pd.Timestamp(dt)})
    cf = pd.DataFrame(clean, columns=["title", "dt"])
    cf["dt"] = pd.to_datetime(cf["dt"])
    risk = select_risk_headlines(cf, n=ri.ctx.settings.n_queries) if len(cf) else []
    cats: dict[str, list[str]] = {}
    for t in risk:
        for c in categorize(t):
            cats.setdefault(c, []).append(t)
    data = {"source": source, "n_headlines": len(cf), "risk_headlines": risk, "categories": cats,
            "flagged_injection": [f["title"] for f in flagged], "aggregates": backup_aggregates,
            "recent": cf.sort_values("dt", ascending=False)["title"].head(15).tolist()}
    ev = []
    if not backup_aggregates or len(cf):
        # headline counts are evidence only when headline text was actually searched
        ev += [("news_n", len(cf), f"Headlines in the last {days} days ({source})", str(len(cf))),
               ("news_risk_n", len(risk), "Risk headlines selected as queries", str(len(risk)))]
    for c, ts in cats.items():
        ev.append((f"news_cat_{c}", len(ts), f"Risk headlines about {c}", str(len(ts))))
    if backup_aggregates:
        ev.append(("news_negative_z", backup_aggregates["news_negative_z"], "Negative-news count z-score (90d)",
                   signed(backup_aggregates["news_negative_z"])))
    s = (f"{source}: " + (f"{len(cf)} headlines, {len(risk)} risk headlines" if not backup_aggregates or len(cf)
                          else "no headline text, counts only")
         + (f" ({', '.join(f'{k} {len(v)}' for k, v in cats.items())})" if cats else "")
         + (f"; {len(flagged)} flagged as possible prompt injection and quarantined" if flagged else "")
         + (f"; negative-news z {signed(backup_aggregates['news_negative_z'])}" if backup_aggregates else ""))
    return data, ev, s


def t_retrieve_precedents(ri: RunInputs, queries: list[str] | None = None, **_):
    st = ri.ctx.settings
    found = ri.ctx.cards.search(list(queries or []), ri.mkt, ri.asof, k=st.k_cards, sem_floor=st.sem_floor,
                                cond_floor=st.cond_floor, min_age_days=st.precedent_min_age_days)
    prec, rej, ev = [], [], []
    for c in (found.itertuples() if len(found) else []):
        item = {"card": c.id, "title": c.title, "date": str(c.date.date()), "tags": list(c.tags),
                "semantic_sim": round(float(c.sem_sim), 3), "condition_sim": round(float(c.cond_sim), 3),
                "matched_headline": c.best_query,
                "conditions_then": {"rv7": _f(c.rv7), "rv7_percentile": _f(c.rv7_pct)},
                "outcome": {"next_7d_rv": _f(c.fwd_rv7), "next_7d_regime": c.next_week, "next_30d_rv": _f(c.fwd_rv30)},
                "evidence": ri.ctx.news.evidence_for(c.title + ". " + c.text, c.date) if ri.ctx.news else [],
                "reject_reason": c.reject_reason}
        if c.qualifies:
            prec.append(item)
            ev.append((f"precedent_{c.id}", item, f"Precedent: {c.title} ({item['date']})",
                       f"{c.title} (semantic {c.sem_sim:.2f}, condition {c.cond_sim:.2f}; next 7d rv "
                       f"{pct(c.fwd_rv7)}, {c.next_week})"))
        else:
            rej.append(item)
    data = {"precedents": prec, "rejected": rej, "embedder": ri.ctx.cards.embedder_name, "n_queries": len(queries or [])}
    ev.append(("precedents_n", len(prec), "Precedents passing the retrieval gate", str(len(prec))))
    s = (f"{len(prec)} precedent(s) passed the gate"
         + (": " + "; ".join(f"{p['title']} (sem {p['semantic_sim']:.2f}, cond {p['condition_sim']:.2f})" for p in prec) if prec else "")
         + (f"; rejected {len(rej)}: " + "; ".join(f"{r['card']} ({r['reject_reason']})" for r in rej) if rej else ""))
    if not queries:
        s = "no risk headlines to search with; no precedents considered"
    return data, ev, s


def t_memory_search(ri: RunInputs, **_):
    res = ri.ctx.memory.search_similar(ri.fingerprint(), ri.asof)
    ev = [("memory_n_scored", res.get("n_scored", 0), "Scored past forecasts available", str(res.get("n_scored", 0)))]
    return res, ev, res["calibration_note"] + (
        " Similar: " + "; ".join(f"{s['asof']} {s['forecast']}/{s['warning']} -> {s['outcome']}" for s in res["similar"])
        if res.get("similar") else "")


TOOLS: dict[str, Callable] = {
    "har_forecast": t_har_forecast,
    "market_snapshot": t_market_snapshot,
    "funding_rates": t_funding_rates,
    "vix_fear": t_vix_fear,
    "stablecoin_peg": t_stablecoin_peg,
    "history_query": t_history_query,
    "news_search": t_news_search,
    "retrieve_precedents": t_retrieve_precedents,
    "memory_search": t_memory_search,
}
READ_ONLY_ALLOW_LIST = frozenset(TOOLS)


class ToolRegistry:
    def __init__(self, ri: RunInputs, state: RunState, max_calls: int, time_limit_s: float):
        self.ri, self.state = ri, state
        self.max_calls, self.deadline = max_calls, state.started + time_limit_s
        self.calls = 0
        self.cache: dict[str, Observation] = {}
        self._lock = threading.Lock()

    def remaining(self) -> int:
        return self.max_calls - self.calls

    def time_left(self) -> float:
        return self.deadline - time.time()

    def call(self, agent: str, tool: str, **args) -> Observation:
        key = tool + json.dumps(args, sort_keys=True, default=str)
        if tool not in READ_ONLY_ALLOW_LIST:
            obs = Observation(tool, False, f"tool {tool!r} is not on the read-only allow-list", error="not allowed")
            self.state.record(agent, "error", obs.summary, tool=tool)
            return obs
        with self._lock:
            if key in self.cache:
                cached = self.cache[key]
                obs = Observation(cached.tool, cached.ok, cached.summary, cached.data, cached.evidence, cached.error, True)
                self.state.record(agent, "observe", f"(cached) {obs.summary}", list(obs.evidence.values()), tool)
                return obs
            if self.calls >= self.max_calls:
                obs = Observation(tool, False, f"tool budget exhausted ({self.max_calls} calls)", error="budget")
                self.state.record(agent, "error", obs.summary, tool=tool)
                return obs
            if time.time() > self.deadline:
                obs = Observation(tool, False, "time limit reached", error="deadline")
                self.state.record(agent, "error", obs.summary, tool=tool)
                return obs
            self.calls += 1
        arg_s = ", ".join(f"{k}={v!r}" if not isinstance(v, list) else f"{k}=[{len(v)} items]" for k, v in args.items())
        self.state.record(agent, "act", f"{tool}({arg_s})", tool=tool)
        try:
            tag = f"{tool}:{args.get('feed', '')}".rstrip(":")
            if tool in self.ri.simulate_failures or tag in self.ri.simulate_failures:
                raise TimeoutError("simulated feed timeout (demo of recovery)")
            data, ev, summary = TOOLS[tool](self.ri, **args)
            evid = {}
            for k, v, label, disp in ev:
                evid[k] = self.state.add_evidence(tool, k, v, label, disp, str(self.ri.asof.date()), agent)
            obs = Observation(tool, True, summary, data, evid)
            self.state.record(agent, "observe", summary, list(evid.values()), tool)
        except Exception as e:  # noqa: BLE001 - errors are observations
            obs = Observation(tool, False, f"{type(e).__name__}: {e}"[:300], error=type(e).__name__)
            self.state.record(agent, "error", obs.summary, tool=tool)
        with self._lock:
            if obs.ok:
                self.cache[key] = obs
        return obs
