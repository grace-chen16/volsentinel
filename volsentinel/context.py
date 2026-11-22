"""Loads the data sources once per process and hands them to each run."""
from __future__ import annotations

import logging
import threading
import time

import pandas as pd

from .config import SETTINGS, Settings
from .data.cards import CardStore
from .data.market import build_features, last_complete_date, load_raw
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
        self.store = StateStore(settings.runs_dir)
        self.llm = LLM(settings)
        self._live_cache: tuple[float, pd.DataFrame, dict] | None = None
        self._live_lock = threading.Lock()
        log.info("context ready in %.1fs (market to %s, news=%s, llm=%s)", time.time() - t0,
                 self.default_asof.date(), bool(self.news), self.llm.available)

    def live_market(self) -> tuple[pd.DataFrame, dict]:
        """Project DB extended with live feeds up to today (cached for 10 minutes)."""
        from .data.live import live_tail
        with self._live_lock:
            if self._live_cache and time.time() - self._live_cache[0] < 600:
                return self._live_cache[1], self._live_cache[2]
            last = self.raw.dropna(subset=["BTC"]).index.max()
            tail, status = live_tail(last)
            raw = pd.concat([self.raw, tail]).sort_index()
            raw = raw[~raw.index.duplicated(keep="last")]
            mkt = build_features(raw, self.settings.regime_window_days, self.settings.p_calm, self.settings.p_stress)
            self._live_cache = (time.time(), mkt, status)
            return mkt, status

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
        }
