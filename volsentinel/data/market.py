"""Market features, regime labels and the HAR forecast.

Adapted from crypto-vol-api/agent/retrieval/market.py (Checkpoint 3 prototype) and
extended with the signals the Tree-of-Thought evaluator checks (funding z-score,
VIX shock, stablecoin peg, drawdown, news-flow aggregates).

Everything uses information available on or before the as-of date. Regimes are
rolling percentiles of 7-day realized vol (Calm < p40, Stressed > p85 of the
trailing two years), so the labels stay meaningful as Bitcoin's vol level drifts.
"""
from __future__ import annotations

import sqlite3
from math import erf, log, sqrt

import numpy as np
import pandas as pd

ANN = np.sqrt(365)
REGIMES = ["Calm", "Elevated", "Stressed"]
STABLES = ["USDT", "USDC", "DAI"]


def load_raw(db_path) -> pd.DataFrame:
    """Raw daily columns from the project database (read-only connection)."""
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        px = pd.read_sql(f"select date, BTC, {', '.join(STABLES)} from crypto_px", con, parse_dates=["date"])
        m = pd.read_sql(
            "select date, BTC_funding_mean, VIX_Close, fear_index_value, news_negative, news_volume, SPX_Close "
            "from master_dataset_pred", con, parse_dates=["date"])
    finally:
        con.close()
    px["date"] = px["date"].dt.normalize()
    m["date"] = m["date"].dt.normalize()
    df = px.set_index("date").join(m.set_index("date"), how="left")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[df.index >= "2017-01-01"].copy()


def build_features(raw: pd.DataFrame, window: int = 730, p_calm: float = 0.40, p_stress: float = 0.85) -> pd.DataFrame:
    df = raw.copy()
    r = np.log(df.BTC).diff()
    df["ret"] = r
    df["rv1"] = r.abs() * ANN
    df["rv7"] = np.sqrt((r ** 2).rolling(7).sum() / 7) * ANN
    df["rv30"] = np.sqrt((r ** 2).rolling(30).sum() / 30) * ANN
    df["fwd_rv7"] = df["rv7"].shift(-7)
    df["fwd_rv30"] = df["rv30"].shift(-30)
    df["thr_calm"] = df["rv7"].rolling(window, min_periods=365).quantile(p_calm)
    df["thr_stress"] = df["rv7"].rolling(window, min_periods=365).quantile(p_stress)
    df["regime_now"] = np.select([df.rv7 < df.thr_calm, df.rv7 > df.thr_stress], [0, 2], 1)
    df["fwd_regime"] = np.select([df.fwd_rv7 < df.thr_calm, df.fwd_rv7 > df.thr_stress], [0, 2], 1)
    df.loc[df.fwd_rv7.isna() | df.thr_calm.isna(), "fwd_regime"] = -1
    # condition fingerprint and ToT signals (all trailing)
    df["rv7_pct"] = df["rv7"].rolling(window, min_periods=365).rank(pct=True)
    df["ret7"] = np.log(df.BTC).diff(7)
    df["ret30"] = np.log(df.BTC).diff(30)
    # z-score funding within each source so live (OKX) and archived rates never share a window
    src = df["funding_src"].fillna("db") if "funding_src" in df else pd.Series("db", index=df.index)
    f = df["BTC_funding_mean"]
    g = f.groupby(src)
    mu = g.transform(lambda s: s.rolling(90, min_periods=30).mean())
    sd = g.transform(lambda s: s.rolling(90, min_periods=30).std())
    df["funding_z"] = (f - mu) / sd
    df["vix"] = df["VIX_Close"].ffill(limit=5)
    df["vix_chg5"] = df["vix"] / df["vix"].shift(5) - 1
    df["fear"] = df["fear_index_value"].ffill(limit=3)
    stab = df[[c for c in STABLES if c in df]].where(lambda x: x > 0)
    df["stable_min"] = stab.min(axis=1).rolling(3, min_periods=1).min()
    for c in ("news_negative", "news_volume"):
        s = df[c]
        df[c + "_z"] = (s - s.rolling(90, min_periods=30).mean()) / s.rolling(90, min_periods=30).std()
    return df


def load_market(db_path, **kw) -> pd.DataFrame:
    return build_features(load_raw(db_path), **kw)


def last_complete_date(df: pd.DataFrame) -> pd.Timestamp:
    """Latest date with price, thresholds and funding all present."""
    ok = df.dropna(subset=["rv7", "thr_calm", "BTC_funding_mean"])
    return ok.index.max()


FP_COLS = ["rv7_pct", "ret30", "funding_z", "vix", "fear"]
FP_SCALE = {"rv7_pct": 0.25, "ret30": 0.15, "funding_z": 1.5, "vix": 8.0, "fear": 20.0}


def condition_similarity(df, asof, dates) -> np.ndarray:
    """Gaussian similarity between the as-of fingerprint and past dates (1 = identical)."""
    a = df.loc[asof, FP_COLS]
    b = df.reindex(pd.DatetimeIndex(dates))[FP_COLS]
    cols = [c for c in FP_COLS if pd.notna(a[c])]
    if not cols:
        return np.zeros(len(b))
    z = sum(((b[c] - a[c]) / FP_SCALE[c]) ** 2 for c in cols)
    z = z.fillna(z.max() if z.notna().any() else 50.0)
    return np.exp(-z / (2 * len(cols))).values


class HAR:
    """HAR on log realized vol: log fwd_rv7 ~ 1 + log rv1 + log rv7 + log rv30.

    Fit on the five years before the as-of date, using only targets whose
    7-day window closed on or before the as-of date (no look-ahead)."""

    def fit(self, df, asof):
        d = df[df.index <= asof - pd.Timedelta(days=7)].dropna(subset=["rv1", "rv7", "rv30", "fwd_rv7"])
        d = d[d.index >= asof - pd.Timedelta(days=5 * 365)]
        if len(d) < 200:
            raise ValueError(f"not enough history to fit HAR before {asof.date()} ({len(d)} rows)")
        X = self._X(d)
        y = np.log(d.fwd_rv7.values)
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.sigma = float(np.std(y - X @ self.beta, ddof=X.shape[1]))
        self.n = len(d)
        return self

    @staticmethod
    def _X(d):
        eps = 1e-4
        return np.column_stack([np.ones(len(d)), np.log(d.rv1 + eps), np.log(d.rv7 + eps), np.log(d.rv30 + eps)])

    def predict(self, df, asof):
        mu = float((self._X(df.loc[[asof]]) @ self.beta)[0])
        return mu, self.sigma


def regime_probs(mu, sigma, thr_calm, thr_stress) -> np.ndarray:
    """P(Calm), P(Elevated), P(Stressed) for a lognormal forecast of next-week rv7."""
    cdf = lambda x: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    pc, ps = cdf(log(thr_calm)), 1 - cdf(log(thr_stress))
    return np.array([pc, max(0.0, 1 - pc - ps), ps])


def naive_probs(df, asof, sigma) -> np.ndarray:
    """Baseline: next week looks like the last week."""
    row = df.loc[asof]
    return regime_probs(np.log(row.rv7), sigma, row.thr_calm, row.thr_stress)


def brier(p, outcome: int) -> float:
    y = np.zeros(3)
    y[outcome] = 1
    return float(((np.asarray(p) - y) ** 2).sum())
