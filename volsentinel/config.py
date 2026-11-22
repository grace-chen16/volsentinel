"""Central settings: data paths, budgets and thresholds from the design checkpoints.

Every threshold is a starting value from the proposals (Checkpoints 3.1, 4.1, 6.1);
they are meant to be tuned on validation weeks only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent                     # .../API Dev/volsentinel
API_DEV_DIR = PROJECT_DIR.parent                 # .../API Dev
COURSE_PROJECT_DIR = API_DEV_DIR.parent          # .../Project


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


@dataclass
class Settings:
    # ---- data -------------------------------------------------------------
    db_path: Path = field(default_factory=lambda: _env_path(
        "VOLSENTINEL_DB", API_DEV_DIR / "crypto-vol-api" / "backend" / "db" / "crypto_data.db"))
    cards_path: Path = field(default_factory=lambda: _env_path(
        "VOLSENTINEL_CARDS", PKG_DIR / "data" / "event_cards.json"))
    news_csv: Path = field(default_factory=lambda: _env_path(
        "VOLSENTINEL_NEWS_CSV", COURSE_PROJECT_DIR / "PredictiveModel" / "RawData" / "CryptoNews" / "news_crypto.csv"))
    cache_dir: Path = field(default_factory=lambda: _env_path("VOLSENTINEL_CACHE", PROJECT_DIR / ".cache"))
    runs_dir: Path = field(default_factory=lambda: _env_path("VOLSENTINEL_RUNS", PROJECT_DIR / "runs"))
    memory_db: Path = field(default_factory=lambda: _env_path("VOLSENTINEL_MEMORY", PROJECT_DIR / "runs" / "experience.db"))
    load_news: bool = field(default_factory=lambda: os.environ.get("VOLSENTINEL_LOAD_NEWS", "1") != "0")

    # ---- budgets (Checkpoints 4.1 / 6.1) ------------------------------------
    max_tool_calls: int = 15
    time_limit_s: float = 90.0
    max_critic_rounds: int = 2
    max_node_evals: int = 11
    max_node_evals_with_tail: int = 14

    # ---- input guardrails ---------------------------------------------------
    max_data_age_hours: float = 24.0

    # ---- retrieval gate (Checkpoint 3.1) --------------------------------------
    sem_floor: float = 0.20
    cond_floor: float = 0.30
    k_cards: int = 3
    n_queries: int = 8
    precedent_min_age_days: int = 30
    blend_weight: float = 0.0           # HAR alone sets probabilities until a blend passes walk-forward test

    # ---- regimes ------------------------------------------------------------
    p_calm: float = 0.40
    p_stress: float = 0.85
    regime_window_days: int = 730

    # ---- Tree of Thought (Checkpoint 4.1) -----------------------------------
    beam_width: int = 2
    depth2_children: int = 2
    prune_below: float = 0.40
    accept_score: float = 0.70
    accept_margin: float = 0.10
    tie_margin: float = 0.05
    unverified_cap: float = 0.50

    # ---- warning gate (fitted on 2021-22 validation weeks by tune.py) ---------
    # Defaults let every warning through; tuned values are read from tuned.json.
    warn_gate_enabled: bool = True
    warn_min_score: float = 0.0
    warn_min_support: float = 0.0
    warn_min_p_stress: float = 0.0
    warn_min_risk_headlines: int = 0
    tuned_source: str = "defaults (untuned)"

    # ---- human review (Checkpoint 6.1) --------------------------------------
    regime_close_margin: float = 0.10
    drift_window_weeks: int = 8

    # ---- optional LLM layer -------------------------------------------------
    llm_model: str = field(default_factory=lambda: os.environ.get("VOLSENTINEL_MODEL", "claude-opus-5-5"))
    llm_enabled: bool = field(default_factory=lambda: os.environ.get("VOLSENTINEL_LLM", "auto") != "off")

    tuned_path: Path = field(default_factory=lambda: _env_path("VOLSENTINEL_TUNED", PKG_DIR / "tuned.json"))

    def __post_init__(self):
        if os.environ.get("VOLSENTINEL_TUNED", "") == "off" or not Path(self.tuned_path).exists():
            return
        try:
            data = json.loads(Path(self.tuned_path).read_text(encoding="utf-8"))
            if not data.get("adopted", False):
                # fitted on validation weeks but not adopted: it did not improve the held-out test
                self.tuned_source = f"defaults ({Path(self.tuned_path).name} fitted but not adopted)"
                return
            for k, v in data.get("params", {}).items():
                if k.startswith("warn_") and hasattr(self, k):
                    setattr(self, k, type(getattr(self, k))(v))
            self.tuned_source = f"{Path(self.tuned_path).name} ({data.get('fitted_on', 'unknown period')})"
        except Exception:  # noqa: BLE001 - a bad file leaves the untuned defaults
            self.tuned_source = f"defaults ({Path(self.tuned_path).name} unreadable)"

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.runs_dir, self.memory_db.parent):
            d.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()

DISCLAIMER = ("VolSentinel is a read-only research tool. This forecast describes volatility risk only; "
              "it is not investment advice and does not recommend buying, selling or holding any asset.")
