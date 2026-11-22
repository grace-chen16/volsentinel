"""Persistent store for live market data.

The project database stays read-only. Every live fetch is merged into a separate
SQLite file (live_data/live_store.db by default) so that:
  * later runs only download the days they do not have yet;
  * replay, evaluation and forecast scoring can use dates after the archive ends;
  * series with a short public history (OKX funding: ~120 days) accumulate over time.

Merge rule: for a date already stored, a newly fetched value replaces the stored one
when it is present; a missing new value never erases a stored one.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

NUM_COLS = ["BTC", "USDT", "USDC", "DAI", "BTC_funding_mean", "VIX_Close", "fear_index_value",
            "news_negative", "news_volume", "SPX_Close"]
TEXT_COLS = ["funding_src"]
SCHEMA = ("create table if not exists live_daily (date text primary key, "
          + ", ".join(f"{c} real" for c in NUM_COLS) + ", "
          + ", ".join(f"{c} text" for c in TEXT_COLS) + ", fetched_at real)")


class LiveStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._con() as c:
            c.execute(SCHEMA)

    def _con(self):
        return sqlite3.connect(self.path, timeout=10)

    def load(self) -> pd.DataFrame:
        with self._con() as c:
            df = pd.read_sql("select * from live_daily order by date", c)
        if df.empty:
            return pd.DataFrame(columns=NUM_COLS + TEXT_COLS, index=pd.DatetimeIndex([], name="date"))
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").drop(columns=["fetched_at"])

    def last_date(self) -> pd.Timestamp | None:
        with self._con() as c:
            d = c.execute("select max(date) from live_daily where BTC is not null").fetchone()[0]
        return pd.Timestamp(d) if d else None

    def upsert(self, rows: pd.DataFrame) -> dict:
        """Merge fetched rows into the store. Returns counts of new and updated dates."""
        if rows is None or rows.empty:
            return {"new": 0, "updated": 0}
        rows = rows.reindex(columns=NUM_COLS + TEXT_COLS)
        rows.index = pd.DatetimeIndex(rows.index).normalize()
        with self._lock:
            old = self.load()
            merged = rows.combine_first(old.reindex(columns=NUM_COLS + TEXT_COLS)).loc[rows.index]
            new = int((~rows.index.isin(old.index)).sum())
            now = time.time()
            recs = [(d.strftime("%Y-%m-%d"), *[None if pd.isna(r[c]) else float(r[c]) for c in NUM_COLS],
                     *[None if pd.isna(r[c]) else str(r[c]) for c in TEXT_COLS], now)
                    for d, r in merged.iterrows()]
            cols = ["date"] + NUM_COLS + TEXT_COLS + ["fetched_at"]
            with self._con() as c:
                c.executemany(f"insert or replace into live_daily ({', '.join(cols)}) "
                              f"values ({', '.join('?' * len(cols))})", recs)
        return {"new": new, "updated": len(rows) - new}

    def summary(self) -> dict:
        df = self.load()
        if df.empty:
            return {"path": str(self.path), "days": 0}
        return {"path": str(self.path), "days": int(len(df)), "first": str(df.index.min().date()),
                "last": str(df.index.max().date()),
                "funding_days": int(df.BTC_funding_mean.notna().sum())}
