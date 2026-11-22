"""Scenario catalog and signal checks for the Tree-of-Thought step (Checkpoint 4.1).

A thought is a 7-day scenario hypothesis: its mechanism, the signals that would
confirm it, the signals that would refute it, and the regime direction it implies
relative to HAR. Signals are checked deterministically against evidence already
in the shared state, so every branch is tested against the same cached tool
outputs.

Signal states: confirmed | not_confirmed | refuted-by-data is handled separately
| unverified (the data was missing or its tool failed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .state import RunState


@dataclass
class SignalResult:
    name: str
    state: str                 # confirmed | not_confirmed | unverified
    text: str                  # human description (no numbers; numbers come from evidence)
    evidence: list[str] = field(default_factory=list)


def _v(st: RunState, key):
    e = st.find_evidence(key)
    return (None, None) if e is None or e.value is None else (e.value, e.id)


def _num_rule(keys: list[str], rule: Callable[..., bool], text: str):
    def check(st: RunState, name: str) -> SignalResult:
        vals, ids = [], []
        for k in keys:
            v, i = _v(st, k)
            if v is None:
                return SignalResult(name, "unverified", text)
            vals.append(v)
            ids.append(i)
        return SignalResult(name, "confirmed" if rule(*vals) else "not_confirmed", text, ids)
    return check


def _news_cat(cats: list[str], text: str):
    def check(st: RunState, name: str) -> SignalResult:
        n, nid = _v(st, "news_n")
        if n is None:
            return SignalResult(name, "unverified", text)
        ids = [e.id for c in cats if (e := st.find_evidence(f"news_cat_{c}")) is not None and e.value]
        return SignalResult(name, "confirmed" if ids else "not_confirmed", text, ids or [nid])
    return check


def _precedent(family: str, text: str):
    def check(st: RunState, name: str) -> SignalResult:
        n, nid = _v(st, "precedents_n")
        if n is None:
            return SignalResult(name, "unverified", text)
        ids = [e.id for e in st.evidence.values()
               if e.key.startswith("precedent_") and isinstance(e.value, dict) and family in e.value.get("tags", [])]
        return SignalResult(name, "confirmed" if ids else "not_confirmed", text, ids or [nid])
    return check


def _hist_elevated(signal: str, text: str):
    def check(st: RunState, name: str) -> SignalResult:
        s, sid = _v(st, f"hist_{signal}_stress")
        b, bid = _v(st, "hist_base_stress")
        if s is None or b is None:
            return SignalResult(name, "unverified", text)
        return SignalResult(name, "confirmed" if s > 1.5 * b and s >= 0.2 else "not_confirmed", text, [sid, bid])
    return check


def _news_quiet(st: RunState, name: str) -> SignalResult:
    text = "few or no risk headlines in the last three days"
    n, nid = _v(st, "news_risk_n")
    if n is None:
        z, zid = _v(st, "news_negative_z")
        if z is None:
            return SignalResult(name, "unverified", text)
        return SignalResult(name, "confirmed" if z < 1 else "not_confirmed", text, [zid])
    crisis = [e.id for c in ("insolvency", "hack", "depeg", "liquidation")
              if (e := st.find_evidence(f"news_cat_{c}")) is not None and e.value]
    return SignalResult(name, "confirmed" if n <= 1 and not crisis else "not_confirmed", text, [nid] + crisis)


def _news_busy(st: RunState, name: str) -> SignalResult:
    text = "a cluster of risk headlines in the last three days"
    n, nid = _v(st, "news_risk_n")
    if n is None:
        z, zid = _v(st, "news_negative_z")
        if z is None:
            return SignalResult(name, "unverified", text)
        return SignalResult(name, "confirmed" if z > 2 else "not_confirmed", text, [zid])
    return SignalResult(name, "confirmed" if n >= 4 else "not_confirmed", text, [nid])


SIGNALS: dict[str, Callable[[RunState, str], SignalResult]] = {
    "funding_high": _num_rule(["funding_z"], lambda z: z > 2, "funding far above its 90-day range (crowded longs)"),
    "funding_low": _num_rule(["funding_z"], lambda z: z < -2, "funding far below its 90-day range (crowded shorts)"),
    "funding_extreme": _num_rule(["funding_z"], lambda z: abs(z) > 2, "funding far outside its 90-day range"),
    "funding_normal": _num_rule(["funding_z"], lambda z: abs(z) <= 1.5, "funding inside its normal range"),
    "vix_high": _num_rule(["vix", "vix_chg5"], lambda v, c: v > 25 or c > 0.20, "equity volatility elevated or jumping"),
    "vix_normal": _num_rule(["vix", "vix_chg5"], lambda v, c: v < 22 and c < 0.15, "equity volatility calm"),
    "fear_extreme": _num_rule(["fear"], lambda f: f < 25, "crypto sentiment in extreme fear"),
    "greed_extreme": _num_rule(["fear"], lambda f: f > 75, "crypto sentiment in extreme greed"),
    "vol_jump": _num_rule(["rv1", "rv7_now"], lambda a, b: a > 2.5 * b, "a one-day move far larger than the past week's"),
    "vol_cooling": _num_rule(["rv1", "rv7_now", "rv7_pct"], lambda a, b, p: a < b and p > 0.6, "volatility high but cooling"),
    "vol_low_pct": _num_rule(["rv7_pct"], lambda p: p < 0.5, "recent volatility low for the past two years"),
    "drawdown": _num_rule(["ret7"], lambda r: r < -0.10, "a sharp drawdown over the past week"),
    "no_drawdown": _num_rule(["ret7"], lambda r: r > -0.03, "no meaningful drawdown over the past week"),
    "stable_depeg": _num_rule(["stable_min"], lambda s: s < 0.985, "a major stablecoin trading below its peg"),
    "stable_pegged": _num_rule(["stable_min"], lambda s: s >= 0.997, "stablecoins holding their peg"),
    "news_insolvency": _news_cat(["insolvency", "hack"], "headlines about withdrawal halts, insolvency or hacks"),
    "news_depeg": _news_cat(["depeg"], "headlines about a stablecoin depeg"),
    "news_liquidation": _news_cat(["liquidation"], "headlines about liquidations or leverage"),
    "news_macro": _news_cat(["macro"], "headlines about rates, inflation, tariffs or equity selloffs"),
    "news_scheduled": _news_cat(["scheduled"], "headlines about a scheduled catalyst (FOMC, CPI, ETF, unlock, vote)"),
    "news_regulation": _news_cat(["regulation"], "headlines about lawsuits, bans or enforcement"),
    "news_quiet": _news_quiet,
    "news_busy": _news_busy,
    "precedent_insolvency": _precedent("insolvency_contagion", "a gate-passing precedent of platform or stablecoin failure"),
    "precedent_leverage": _precedent("leverage_flush", "a gate-passing precedent of a leverage flush"),
    "precedent_macro": _precedent("macro_risk_off", "a gate-passing precedent of a macro risk-off shock"),
    "precedent_scheduled": _precedent("scheduled_event", "a gate-passing precedent of a scheduled or regulatory event"),
    "hist_funding": _hist_elevated("funding_spike", "past funding spikes were often followed by stressed weeks"),
}


def check(st: RunState, name: str) -> SignalResult:
    return SIGNALS[name](st, name)


@dataclass
class Mechanism:
    key: str
    label: str
    predicted: list[str]
    refuting: list[str]
    warning: str               # base warning level: none | watch | warning | severe


@dataclass
class Family:
    key: str
    label: str
    severity: int              # used by the protected tail slot
    direction: str             # calm | up | mild_up  (relative to HAR)
    predicted: list[str]
    refuting: list[str]
    mechanisms: list[Mechanism]
    precedent_signal: str | None


FAMILIES: list[Family] = [
    Family("quiet_drift", "Quiet drift", 0, "calm",
           ["funding_normal", "vix_normal", "news_quiet"], ["funding_extreme", "vix_high", "vol_jump"],
           [Mechanism("signals_stay_normal", "Signals stay normal; the HAR baseline holds",
                      ["funding_normal", "vix_normal", "news_quiet", "no_drawdown"], ["funding_extreme", "vix_high", "drawdown"], "none"),
            Mechanism("post_shock_normalisation", "Post-shock normalisation; volatility decays",
                      ["vol_cooling", "funding_normal", "news_quiet"], ["vol_jump", "drawdown"], "none")],
           None),
    Family("leverage_flush", "Leverage flush", 2, "up",
           ["funding_extreme", "news_liquidation", "vol_jump"], ["funding_normal"],
           [Mechanism("long_squeeze", "Crowded longs unwind through forced liquidations",
                      ["funding_high", "greed_extreme", "news_liquidation", "hist_funding", "precedent_leverage"],
                      ["funding_low", "funding_normal"], "warning"),
            Mechanism("cascade_underway", "A liquidation cascade is already under way",
                      ["vol_jump", "drawdown", "news_liquidation", "fear_extreme"], ["vol_cooling", "no_drawdown"], "warning")],
           "precedent_leverage"),
    Family("insolvency_contagion", "Insolvency contagion", 3, "up",
           ["news_insolvency", "precedent_insolvency", "news_busy"], ["news_quiet"],
           [Mechanism("exchange_run", "A platform run: withdrawals halt and contagion spreads",
                      ["news_insolvency", "precedent_insolvency", "fear_extreme", "drawdown"], ["news_quiet", "greed_extreme"], "severe"),
            Mechanism("stablecoin_depeg", "A stablecoin loses its peg",
                      ["stable_depeg", "news_depeg", "precedent_insolvency"], ["stable_pegged"], "severe")],
           "precedent_insolvency"),
    Family("macro_risk_off", "Macro risk-off", 2, "up",
           ["vix_high", "news_macro", "precedent_macro"], ["vix_normal"],
           [Mechanism("equity_vol_shock", "An equity volatility shock spills into crypto",
                      ["vix_high", "fear_extreme", "drawdown", "precedent_macro"], ["vix_normal"], "warning"),
            Mechanism("policy_shock", "A rates or policy surprise reprices risk assets",
                      ["news_macro", "vix_high", "precedent_macro"], ["vix_normal", "news_quiet"], "watch")],
           "precedent_macro"),
    Family("scheduled_event", "Scheduled event", 1, "mild_up",
           ["news_scheduled", "news_regulation", "precedent_scheduled"], ["news_quiet"],
           [Mechanism("known_catalyst", "A known catalyst (FOMC, CPI, ETF deadline, unlock) moves the market",
                      ["news_scheduled", "precedent_scheduled"], ["news_quiet"], "watch"),
            Mechanism("regulatory_decision", "A regulatory or legal decision lands",
                      ["news_regulation", "precedent_scheduled", "fear_extreme"], ["news_quiet"], "watch")],
           "precedent_scheduled"),
]
FAMILY = {f.key: f for f in FAMILIES}
WARN_LEVELS = ["none", "watch", "warning", "severe"]
