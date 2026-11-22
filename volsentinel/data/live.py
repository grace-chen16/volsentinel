"""Read-only live data feeds (no API keys, no exchange accounts).

Used in live mode to extend the project database up to today:
  * BTC, USDT and DAI daily closes  - Coinbase Exchange public candles
  * BTC perpetual funding rate      - OKX public funding-rate history
  * VIX                             - Yahoo Finance chart endpoint
  * Fear & Greed                    - alternative.me
  * headlines (primary / backup)    - Cointelegraph RSS / CoinDesk RSS
Every fetch has a short timeout; callers turn failures into observations.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "VolSentinel/0.1 (read-only research client)"}
TIMEOUT = 10


def _get(url, params=None):
    r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def coinbase_daily(product: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily closes (UTC) for a Coinbase product, paging 300 candles at a time."""
    out = {}
    s = start.normalize()
    while s <= end:
        e = min(s + pd.Timedelta(days=299), end)
        rows = _get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                    {"granularity": 86400, "start": s.isoformat(), "end": (e + pd.Timedelta(days=1)).isoformat()}).json()
        for t, _lo, _hi, _op, close, _vol in rows:
            out[pd.Timestamp(t, unit="s").normalize()] = float(close)
        s = e + pd.Timedelta(days=1)
    return pd.Series(out, dtype=float).sort_index()


def okx_funding_daily(days: int = 120) -> pd.Series:
    """Daily mean of the 8-hourly BTC-USDT-SWAP funding rate."""
    recs, after = [], None
    while len(recs) < days * 3:
        p = {"instId": "BTC-USDT-SWAP", "limit": 100}
        if after:
            p["after"] = after
        data = _get("https://www.okx.com/api/v5/public/funding-rate-history", p).json().get("data", [])
        if not data:
            break
        recs += data
        after = data[-1]["fundingTime"]
    if not recs:
        raise RuntimeError("OKX returned no funding data")
    s = pd.Series({pd.Timestamp(int(r["fundingTime"]), unit="ms"): float(r["fundingRate"]) for r in recs})
    return s.groupby(s.index.normalize()).mean().sort_index()


def yahoo_vix(range_: str = "2y") -> pd.Series:
    j = _get("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX", {"range": range_, "interval": "1d"}).json()
    res = j["chart"]["result"][0]
    ts = [pd.Timestamp(t, unit="s").normalize() for t in res["timestamp"]]
    return pd.Series(res["indicators"]["quote"][0]["close"], index=ts, dtype=float).dropna()


def fear_greed(limit: int = 800) -> pd.Series:
    data = _get("https://api.alternative.me/fng/", {"limit": limit}).json()["data"]
    return pd.Series({pd.Timestamp(int(d["timestamp"]), unit="s").normalize(): float(d["value"]) for d in data}).sort_index()


def live_tail(after: pd.Timestamp, today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Raw rows for dates after `after`, in the same columns as data.market.load_raw.
    Returns (frame, status) where status maps each feed to 'ok' or its error."""
    today = (today or pd.Timestamp.utcnow().tz_localize(None)).normalize()
    start = after + pd.Timedelta(days=1)
    status, cols = {}, {}
    for name, fn in {
        "BTC": lambda: coinbase_daily("BTC-USD", start, today),
        "USDT": lambda: coinbase_daily("USDT-USD", start, today),
        "DAI": lambda: coinbase_daily("DAI-USD", start, today),
        "BTC_funding_mean": lambda: okx_funding_daily(),
        "VIX_Close": lambda: yahoo_vix(),
        "fear_index_value": lambda: fear_greed(),
    }.items():
        try:
            cols[name] = fn()
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001 - every failure becomes an observation
            status[name] = f"error: {type(e).__name__}: {e}"[:200]
    if "BTC" not in cols or cols["BTC"].empty:
        raise RuntimeError(f"live BTC price unavailable ({status.get('BTC')})")
    idx = cols["BTC"].index
    df = pd.DataFrame(index=idx)
    for k, s in cols.items():
        df[k] = s.reindex(idx)
    for k in ["USDT", "USDC", "DAI", "BTC_funding_mean", "VIX_Close", "fear_index_value",
              "news_negative", "news_volume", "SPX_Close"]:
        if k not in df:
            df[k] = np.nan
    df["funding_src"] = "okx"
    # today's UTC candle is still open: only completed days are used
    return df[(df.index > after) & (df.index < today)], status


def rss_headlines(feed: str = "primary") -> pd.DataFrame:
    url = {"primary": "https://cointelegraph.com/rss",
           "backup": "https://www.coindesk.com/arc/outboundfeeds/rss/"}[feed]
    root = ET.fromstring(_get(url).content)
    rows = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        try:
            dt = parsedate_to_datetime(pub).astimezone(tz=None).replace(tzinfo=None) if pub else None
        except Exception:  # noqa: BLE001
            dt = None
        if title and dt is not None:
            rows.append({"title": title, "dt": pd.Timestamp(dt), "sourceDomain": url.split("/")[2]})
    df = pd.DataFrame(rows, columns=["title", "dt", "sourceDomain"])
    df["date"] = df.dt.dt.normalize() if len(df) else pd.Series(dtype="datetime64[ns]")
    return df
