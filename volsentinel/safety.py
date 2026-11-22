"""Guardrails from the safety plan (Checkpoint 6.1).

Input:  freshness and range checks; news text is data, never instructions.
Output: no buy/sell language, every number traced to a tool output, disclaimer.
Review: the five human-intervention triggers.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- input: news text
INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)\s+"
    r"(instructions?|prompts?|rules?)|system\s*prompt|you\s+are\s+now|act\s+as\s+|new\s+instructions?|"
    r"</?\s*(script|system|assistant|instruction)[^>]*>|\bassistant\s*:|\bsystem\s*:|jailbreak|"
    r"do\s+not\s+follow|reveal\s+(your|the)\s+(prompt|instructions)", re.I)
TAG_RE = re.compile(r"<[^>]{0,200}>")


def sanitize_text(text: str, max_len: int = 300) -> tuple[str, bool]:
    """Return (clean text, injection_flag). Clean text is plain, single-line, bounded.
    Flagged headlines are kept in the trace as data but never used as queries."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "C")
    flagged = bool(INJECTION_RE.search(t))
    t = TAG_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()[:max_len]
    return t, flagged


# ---------------------------------------------------------------- input: user question
ADVICE_RE = re.compile(
    r"\b(should\s+i\s+(buy|sell|short|long|hold|exit|enter|hedge|close|open|add|trim|dump)|"
    r"(buy|sell)\s+or\s+(sell|buy|hold)|what\s+(coin|token|crypto)s?\s+(should|to)\s+(i\s+)?buy|"
    r"price\s+target|when\s+(should|do)\s+i\s+(buy|sell)|how\s+much\s+(should|do)\s+i\s+(buy|sell|invest|put)|"
    r"is\s+it\s+(a\s+)?(good|right)\s+time\s+to\s+(buy|sell)|tell\s+me\s+to\s+(buy|sell)|"
    r"place\s+(a\s+)?(trade|order)|execute\s+(a\s+)?trade)\b", re.I)


def is_advice_request(question: str) -> bool:
    return bool(ADVICE_RE.search(question or ""))


ADVICE_DECLINE = ("VolSentinel does not give personal trading advice or place trades. It can describe "
                  "volatility risk for the coming week; for decisions about your own positions, please "
                  "consult a licensed financial adviser.")

# ---------------------------------------------------------------- output language
TRADE_LANG_RE = re.compile(
    r"\b(you\s+should|we\s+recommend|i\s+recommend|recommend(?:ed|s)?\s+(?:buying|selling|shorting)|"
    r"buy\s+now|sell\s+now|go\s+long|go\s+short|open\s+a\s+(?:long|short)|close\s+your|take\s+profits?|"
    r"stop[- ]loss|price\s+target|(?<![\w-])(?:buy|sell)(?![\w-]))\b", re.I)
FIRM_INSOLVENT_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.]{1,30})\s+(?:is|was|are|will\s+be|appears|looks|seems)\s+(?:to\s+be\s+)?"
    r"(?:insolvent|bankrupt|going\s+under|collapsing)\b")


def trade_language(text: str) -> list[str]:
    return sorted({m.group(0) for m in TRADE_LANG_RE.finditer(text or "")})


def names_firm_insolvent(text: str) -> list[str]:
    return sorted({m.group(1) for m in FIRM_INSOLVENT_RE.finditer(text or "")})


NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")


def numbers_in(text: str) -> list[str]:
    """Numeric tokens in authored text, ignoring years and the evidence ids [E12]."""
    t = re.sub(r"\[E\d+(?:,\s*E\d+)*\]", " ", text or "").replace(",", "")
    t = re.sub(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", " ", t)          # dates
    t = re.sub(r"\b(?:19|20)\d{2}\b", " ", t)                     # years
    t = re.sub(r"\b\d+-(?:day|week|month|year)s?\b|\bday \d\b|\btop \d\b|\b\d+ (?:days?|weeks?|months?|years?|hours?)\b",
               " ", t, flags=re.I)
    return [n.lstrip("+-") for n in NUM_RE.findall(t)]


# ---------------------------------------------------------------- input: market data
RANGES = {"BTC": (100, 10_000_000), "rv7": (0.01, 6.0), "vix": (5, 90), "fear": (0, 100),
          "funding_z": (-15, 15), "stable_min": (0.5, 1.5)}


def validate_market(df: pd.DataFrame, asof: pd.Timestamp, reference: pd.Timestamp, max_age_h: float) -> list[dict]:
    """Freshness and range checks for the inputs at `asof`.

    `reference` is the time the answer is for: now (live) or the as-of close (replay).
    A feed whose last observation is older than max_age_h is stale."""
    issues = []
    upto = df[df.index <= asof]
    feeds = {"BTC": "BTC", "funding": "BTC_funding_mean", "VIX": "VIX_Close", "fear_greed": "fear_index_value"}
    for name, col in feeds.items():
        s = upto[col].dropna() if col in upto else pd.Series(dtype=float)
        if s.empty:
            issues.append({"feed": name, "issue": "missing", "severity": "high" if name in ("BTC", "funding") else "medium"})
            continue
        # daily closes: a value dated D is known at the end of D
        age_h = (reference - (s.index.max() + pd.Timedelta(days=1))).total_seconds() / 3600
        allowance = max_age_h + (72 if name == "VIX" else 0)   # equities close at weekends/holidays
        if age_h > allowance:
            issues.append({"feed": name, "issue": f"stale ({age_h:.0f}h old, limit {allowance:.0f}h)",
                           "severity": "high" if name in ("BTC", "funding") else "medium"})
    if asof in df.index:
        row = df.loc[asof]
        for k, (lo, hi) in RANGES.items():
            v = row.get(k)
            if v is not None and pd.notna(v) and not (lo <= float(v) <= hi):
                issues.append({"feed": k, "issue": f"out of valid range ({v:.4g} not in [{lo}, {hi}])", "severity": "high"})
    return issues


def outside_two_year_range(df: pd.DataFrame, asof: pd.Timestamp, cols=("rv7", "funding_z", "vix", "ret30")) -> list[str]:
    hist = df[(df.index < asof) & (df.index >= asof - pd.Timedelta(days=730))]
    out = []
    for c in cols:
        v = df.at[asof, c] if asof in df.index else np.nan
        h = hist[c].dropna()
        if pd.notna(v) and len(h) > 100 and (v < h.min() or v > h.max()):
            out.append(f"{c}={v:.3g} outside past-2y range [{h.min():.3g}, {h.max():.3g}]")
    return out
