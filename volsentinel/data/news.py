"""News archive, embeddings and risk-headline selection.

Adapted from crypto-vol-api/agent/retrieval/news_store.py. One headline is one
chunk. Embeddings are latent semantic analysis (TF-IDF + truncated SVD, 256-d)
fitted on the archive; set VOLSENTINEL_EMBEDDER=st to use a sentence-transformer
when its model is available locally. The fitted embedder and the vectors are
cached so the server starts in seconds after the first run.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DIM = 256

SOCIAL = {"twitter.com", "reddit.com", "square.binance.com", "youtube.com", "t.me"}

RISK_RE = re.compile(
    r"\b(?:withdrawals?|halts?|halted|paus(?:e|es|ed)|insolven\w*|bankrupt\w*|hack(?:s|ed|er|ers)?|exploit(?:s|ed)?|"
    r"liquidat\w*|de-?peg\w*|collaps\w*|crash\w*|outflows?|contagion|default\w*|lawsuit|sues|sued|charged|"
    r"seiz\w*|bans?|banned|plung\w*|tumbl\w*|sell-?off|margin call)\b", re.I)

# Headline categories used as ToT evidence. Matching is on wording only.
CATEGORIES = {
    "insolvency": r"\b(withdrawals?\s+(?:halt|paus|suspend|freez)\w*|halts?\s+withdrawals?|paus\w*\s+withdrawals?|insolven\w*|bankrupt\w*|bank run|chapter 11|outflows?)\b",
    "depeg": r"\b(de-?peg\w*|(?:stablecoin|usd[tc]|tether|ust|dai)\w*(?:\s+\w+){0,2}\s+(?:loses?|lost|below|slip\w*|"
             r"break\w*|plung\w*|tumbl\w*|crash\w*|falls?|drops?)|lost (?:its|the) peg)\b",
    "hack": r"\b(hack(?:s|ed|er|ers)?|exploit(?:s|ed)?|drained|stolen)\b",
    "liquidation": r"\b(liquidat\w*|margin call|leverage\w*|short squeeze|long squeeze|open interest)\b",
    "regulation": r"\b(sec\b|sues|sued|lawsuit|charged|crackdown|bans?|banned|regulat\w*|seiz\w*)\b",
    "macro": r"\b(fed\b|fomc|rate (?:hike|cut)s?|cpi|inflation|tariff\w*|recession|treasury yields?|stocks? (?:plunge|tumble|sell)\w*|risk-off)\b",
    "scheduled": r"\b(fomc|cpi|etf (?:deadline|decision|approval)|halving|unlock\w*|expir\w*|election|hearing|vote)\b",
}
CAT_RE = {k: re.compile(v, re.I) for k, v in CATEGORIES.items()}


def categorize(title: str) -> list[str]:
    return [k for k, rx in CAT_RE.items() if rx.search(title or "")]


def load_news(csv_path, start="2019-01-01", end="2026-12-31") -> pd.DataFrame:
    cols = ["id", "title", "description", "sourceDomain", "newsDatetime"]
    df = pd.read_csv(csv_path, usecols=cols, na_values=["NULL"]).drop_duplicates("id")
    df["dt"] = pd.to_datetime(df.newsDatetime, errors="coerce")
    df = df[(df.dt >= start) & (df.dt < end) & df.title.notna()].copy()
    df["title"] = df.title.astype(str)
    # news-site headlines only: social posts are short, promotional and embed poorly
    df = df[~df.sourceDomain.isin(SOCIAL) & (df.title.str.len() >= 25)
            & ~df.title.str.contains(r"^RT @|@\w", regex=True, na=False)].copy()
    df["date"] = df.dt.dt.normalize()
    df["text"] = (df.title.fillna("") + ". " + df.description.fillna("").str.slice(0, 300)).str.strip(". ")
    return df.sort_values("dt").reset_index(drop=True)


class Embedder:
    def __init__(self, backend: str | None = None):
        backend = backend or os.environ.get("VOLSENTINEL_EMBEDDER", "lsa")
        self.backend = backend
        if backend == "st":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit(self, texts):
        if self.backend == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = list(texts)
            min_df = 3 if len(texts) > 1000 else 1
            self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df,
                                         max_features=200_000, sublinear_tf=True)
            X = self.tfidf.fit_transform(texts)
            dim = min(DIM, X.shape[1] - 1, X.shape[0] - 1)
            self.svd = TruncatedSVD(max(2, dim), random_state=0).fit(X)
        return self

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        if self.backend == "st":
            v = self.model.encode(texts, batch_size=256, show_progress_bar=False)
        else:
            v = self.svd.transform(self.tfidf.transform(texts))
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("model", None)
        return d


class NewsStore:
    """In-memory vector store over the archive (numpy; swap for FAISS at scale)."""

    def __init__(self, news: pd.DataFrame, embedder: Embedder, vecs: np.ndarray):
        self.news = news
        self.emb = embedder
        self.vecs = vecs.astype(np.float32)

    @classmethod
    def build(cls, csv_path: Path, cache_dir: Path) -> "NewsStore":
        st = os.stat(csv_path)
        key = hashlib.md5(f"{csv_path}|{st.st_size}|{st.st_mtime}|v1".encode()).hexdigest()[:12]
        cache = Path(cache_dir) / f"news_{key}.pkl"
        if cache.exists():
            with open(cache, "rb") as fh:
                news, emb, vecs = pickle.load(fh)
            log.info("news archive loaded from cache (%d headlines)", len(news))
            return cls(news, emb, vecs)
        log.info("building news index from %s (first run only)", csv_path)
        news = load_news(csv_path)
        emb = Embedder("lsa").fit(news.text.values)
        vecs = emb.encode(news.text.values).astype(np.float16)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump((news, emb, vecs), fh, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(news, emb, vecs)

    def window(self, asof: pd.Timestamp, days: int = 3) -> pd.DataFrame:
        """Headlines dated in (asof - days, asof]."""
        n = self.news
        return n[(n.date > asof - pd.Timedelta(days=days)) & (n.date <= asof)]

    def evidence_for(self, text: str, date: pd.Timestamp, n: int = 2) -> list[tuple[str, str]]:
        """Dated archive headlines from the week of a past episode that best match it."""
        lo, hi = date - pd.Timedelta(days=1), date + pd.Timedelta(days=2)
        idx = np.where(((self.news.date >= lo) & (self.news.date <= hi)).values)[0]
        if len(idx) == 0:
            return []
        q = self.emb.encode([text])[0]
        best = idx[np.argsort(-(self.vecs[idx] @ q))[:n]]
        return [(self.news.date.iloc[i].date().isoformat(), str(self.news.title.iloc[i])) for i in best]


def select_risk_headlines(headlines: pd.DataFrame, n: int = 8) -> list[str]:
    """Up to n risk headlines, newest first. Selection uses wording only: the archive's
    reader-vote counts were tallied later and would leak future information."""
    if headlines is None or len(headlines) == 0:
        return []
    risk = headlines[headlines.title.str.contains(RISK_RE)].drop_duplicates("title")
    return list(risk.sort_values("dt", ascending=False).title.head(n).values)
