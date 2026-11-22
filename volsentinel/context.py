"""Loads the data sources once per process and hands them to each run."""
from __future__ import annotations

import logging
import threading
import time

import pandas as pd

from .config import SETTINGS, Settings
from .data.cards import CardStore
from .data.market import build_features, last_complete_date, load_raw
from .data.live_store import LiveStore
from .data.news import NewsStore
from .llm import LLM
from .memory import ExperienceMemory
from .state import StateStore

log = logging.getLogger(__name__)


class Context:
    def __init__(self, settings: Settings = SETTINGS, load_news: bool | None = None):
        self.settings = settings
        settings.ensure_dirs()
        t0 = time.time()
        self.raw = load_raw(settings.db_path)
        self.mkt = build_features(self.raw, settings.regime_window_days, settings.p_calm, settings.p_stress)
        self.default_asof = last_complete_date(self.mkt)
        self.news: NewsStore | None = None
        self.news_error: str | None = None
        want_news = settings.load_news if load_news is None else load_news
        if want_news:
            try:
                self.news = NewsStore.build(settings.news_csv, settings.cache_dir)
            except Exception as e:  # noqa: BLE001
                self.news_error = f"{type(e).__name__}: {e}"
                log.warning("news archive unavailable: %s", self.news_error)
        else:
            self.news_error = "news archive disabled by configuration"
        self.cards = CardStore(settings.cards_path, self.mkt, self.news.emb if self.news else None)
        self.memory = ExperienceMemory(settings.memory_db)
        self.live_store = LiveStore(settings.live_store)
        self.store = StateStore(settings.runs_dir)
        self.llm = LLM(settings)
        self._live_cache: tuple[float, pd.DataFrame, dict] | None = None
        self._live_lock = threading.Lock()
        log.info("context ready in %.1fs (market to %s, news=%s, llm=%s)", time.time() - t0,
                 self.default_asof.date(), bool(self.news), self.llm.available)

    # ------------------------------------------------------------------ live data
    def _features(self, raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw, self.settings.regime_window_days, self.settings.p_calm, self.settings.p_stress)

    def _with_stored(self) -> pd.DataFrame:
        """Archive rows, then stored live rows for the dates after the archive's last BTC price."""
        stored = self.live_store.load()
        db_last = self.raw.dropna(subset=["BTC"]).index.max()
        stored = stored[stored.index > db_last].dropna(axis=1, how="all")
        if stored.empty:
            return self.raw
        raw = pd.concat([self.raw, stored]).sort_index()
        return raw[~raw.index.duplicated(keep="last")]

    def stored_market(self) -> pd.DataFrame:
        """Archive plus every stored live day, without any network call. Used for scoring."""
        return self._features(self._with_stored())

    def ingest(self) -> dict:
        """Fetch live feeds from the last stored day onward and merge them into the live store."""
        from .data.live import live_tail
        db_last = self.raw.dropna(subset=["BTC"]).index.max()
        store_last = self.live_store.last_date()
        after = db_last
        if store_last is not None and store_last > db_last:
            after = max(db_last, store_last - pd.Timedelta(days=self.settings.live_overlap_days))
        tail, status = live_tail(after)
        counts = self.live_store.upsert(tail)
        status["stored"] = f"{counts['new']} new day(s), {counts['updated']} updated"
        return {"fetched_from": str((after + pd.Timedelta(days=1)).date()), "counts": counts,
                "status": status, "store": self.live_store.summary()}

    def live_market(self, force: bool = False) -> tuple[pd.DataFrame, dict]:
        """Archive + stored live days + a fresh fetch, persisted to the live store
        (cached for 10 minutes in a running server)."""
        with self._live_lock:
            if not force and self._live_cache and time.time() - self._live_cache[0] < 600:
                return self._live_cache[1], self._live_cache[2]
            res = self.ingest()
            mkt = self._features(self._with_stored())
            self._live_cache = (time.time(), mkt, res["status"])
            return mkt, res["status"]

    def describe(self) -> dict:
        return {
            "market_data_to": str(self.default_asof.date()),
            "market_rows": int(len(self.mkt)),
            "news_archive": (f"{len(self.news.news):,} headlines "
                             f"({self.news.news.date.min().date()} to {self.news.news.date.max().date()})"
                             if self.news else f"unavailable ({self.news_error})"),
            "event_cards": int(len(self.cards.cards)),
            "card_embedder": self.cards.embedder_name,
            "llm": self.llm.describe(),
            "memory_db": str(self.settings.memory_db),
            "live_store": self.live_store.summary(),
        }
