"""Event-card knowledge base and the gated precedent search (Checkpoint 3.1).

Cards describe trigger, mechanism and who was affected; they never carry outcomes.
Outcomes (next 7/30-day realized vol and regime) are computed from price data.

Search for an as-of date T:
  1. score each card by the mean cosine similarity of its 3 best-matching queries;
  2. only cards whose 30-day outcome window closed before T are eligible;
  3. gate: semantic >= sem_floor AND market-condition similarity >= cond_floor;
  4. return top-k with outcome windows and dated archive evidence, plus rejections.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .market import REGIMES, condition_similarity
from .news import Embedder

# Which Tree-of-Thought scenario family each episode illustrates.
CARD_TAGS = {
    "covid-crash": ["macro_risk_off", "leverage_flush"],
    "china-tesla-may21": ["leverage_flush", "scheduled_event"],
    "china-ban-sep21": ["scheduled_event"],
    "el-salvador-sep21": ["leverage_flush"],
    "dec21-flash": ["leverage_flush"],
    "fed-jan22": ["macro_risk_off"],
    "terra-luna": ["insolvency_contagion"],
    "celsius-3ac": ["insolvency_contagion"],
    "merge": ["scheduled_event"],
    "ftx": ["insolvency_contagion"],
    "genesis": ["insolvency_contagion"],
    "svb-usdc": ["insolvency_contagion", "macro_risk_off"],
    "sec-binance-coinbase": ["scheduled_event"],
    "ripple-ruling": ["scheduled_event"],
    "etf-approval": ["scheduled_event"],
    "halving-2024": ["scheduled_event"],
    "mtgox-germany": ["scheduled_event"],
    "carry-unwind": ["macro_risk_off", "leverage_flush"],
    "us-election-2024": ["scheduled_event"],
    "bybit-hack": ["insolvency_contagion"],
    "tariffs-apr25": ["macro_risk_off"],
    "oct25-liquidations": ["leverage_flush", "macro_risk_off"],
}


class CardStore:
    def __init__(self, path, mkt: pd.DataFrame, embedder: Embedder | None = None):
        self.cards = pd.DataFrame(json.load(open(path, encoding="utf-8")))
        self.cards["date"] = pd.to_datetime(self.cards.date)
        self.cards["tags"] = self.cards.id.map(lambda i: CARD_TAGS.get(i, []))
        texts = (self.cards.title + ". " + self.cards.text).values
        if embedder is None:
            # No archive: fit LSA on the cards alone. Similarities are less reliable,
            # so the trace labels the embedder.
            embedder = Embedder("lsa").fit(texts)
            self.embedder_name = "lsa-cards-only"
        else:
            self.embedder_name = f"{embedder.backend}-archive"
        self.emb = embedder
        self.vecs = embedder.encode(texts)
        m = mkt.reindex(self.cards.date)
        for c in ["rv7", "rv7_pct", "fwd_rv7", "fwd_rv30", "fwd_regime"]:
            self.cards[c] = m[c].values

    def search(self, queries, mkt, asof, k=3, sem_floor=0.20, cond_floor=0.30, min_age_days=30) -> pd.DataFrame:
        ok = (self.cards.date <= asof - pd.Timedelta(days=min_age_days)).values
        if not queries or not ok.any():
            return pd.DataFrame()
        S = self.emb.encode(queries) @ self.vecs.T            # queries x cards
        top = np.sort(S, axis=0)[-min(3, len(queries)):]
        out = self.cards.assign(sem_sim=top.mean(axis=0), best_query=[queries[i] for i in S.argmax(axis=0)])
        out = out[ok].sort_values("sem_sim", ascending=False).head(k).copy()
        out["cond_sim"] = condition_similarity(mkt, asof, out.date)
        out["qualifies"] = (out.sem_sim >= sem_floor) & (out.cond_sim >= cond_floor)
        out["reject_reason"] = np.where(
            out.qualifies, "",
            np.where(out.sem_sim < sem_floor,
                     np.where(out.cond_sim < cond_floor, "semantic and condition below floor", "semantic below floor"),
                     "market conditions differ (condition below floor)"))
        out["next_week"] = out.fwd_regime.map(lambda r: REGIMES[int(r)] if pd.notna(r) and r >= 0 else "unknown")
        return out
