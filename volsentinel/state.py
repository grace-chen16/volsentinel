"""Shared state store (the "MCP state manager" of Checkpoints 4.1/5.1).

All agents read and write structured records here instead of passing free text:
  * evidence  - every number a tool produced, with an id (E1, E2, ...) to cite;
  * log       - short-term working memory: each reason / act / observe step;
  * tree      - the Tree-of-Thought nodes with scores and status;
  * messages  - agent-to-agent hand-offs (critic feedback, re-fetch requests).
mcp_server.py exposes the same store read-only over MCP.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


@dataclass
class Evidence:
    id: str
    tool: str
    key: str
    value: Any
    label: str
    display: str            # how the value may appear in text, e.g. "44%"
    data_date: str | None
    agent: str
    ts: float


@dataclass
class LogEntry:
    step: int
    agent: str
    kind: str               # plan | reason | act | observe | note | error | handoff
    text: str
    ts: float
    evidence: list[str] = field(default_factory=list)
    tool: str | None = None


class RunState:
    def __init__(self, question: str, asof: str, mode: str, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.question = question
        self.asof = asof
        self.mode = mode
        self.started = time.time()
        self.evidence: dict[str, Evidence] = {}
        self.log: list[LogEntry] = []
        self.tree: dict[str, dict] = {}
        self.messages: list[dict] = []
        self.meta: dict[str, Any] = {}
        self._lock = threading.RLock()

    # ---- evidence -----------------------------------------------------------
    def add_evidence(self, tool, key, value, label, display=None, data_date=None, agent="") -> str:
        with self._lock:
            eid = f"E{len(self.evidence) + 1}"
            self.evidence[eid] = Evidence(eid, tool, key, _jsonable(value), label,
                                          display if display is not None else str(value),
                                          data_date, agent, time.time() - self.started)
            return eid

    def find_evidence(self, key: str) -> Evidence | None:
        with self._lock:
            for e in reversed(list(self.evidence.values())):
                if e.key == key:
                    return e
        return None

    # ---- log ----------------------------------------------------------------
    def record(self, agent, kind, text, evidence=None, tool=None) -> LogEntry:
        with self._lock:
            e = LogEntry(len(self.log) + 1, agent, kind, text, round(time.time() - self.started, 3),
                         list(evidence or []), tool)
            self.log.append(e)
            return e

    def send(self, sender, to, kind, body):
        with self._lock:
            self.messages.append({"from": sender, "to": to, "kind": kind, "body": _jsonable(body),
                                  "t": round(time.time() - self.started, 3)})
            self.record(sender, "handoff", f"{kind} -> {to}")

    def elapsed(self) -> float:
        return time.time() - self.started

    # ---- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        with self._lock:
            return {
                "run_id": self.run_id, "question": self.question, "asof": self.asof, "mode": self.mode,
                "evidence": {k: asdict(v) for k, v in self.evidence.items()},
                "log": [asdict(e) for e in self.log],
                "tree": _jsonable(self.tree),
                "messages": self.messages,
                "meta": _jsonable(self.meta),
            }


class StateStore:
    """Process-wide registry of runs, persisted as JSON under runs/."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._live: dict[str, RunState] = {}
        self._lock = threading.Lock()

    def new_run(self, question, asof, mode) -> RunState:
        st = RunState(question, asof, mode)
        with self._lock:
            self._live[st.run_id] = st
        return st

    def save(self, st: RunState, answer: dict) -> Path:
        p = self.runs_dir / f"{st.run_id}.json"
        payload = {"answer": _jsonable(answer), "state": st.to_dict()}
        p.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        with self._lock:
            self._live.pop(st.run_id, None)
        return p

    def load(self, run_id: str) -> dict | None:
        if not run_id.replace("-", "").isalnum():
            return None
        with self._lock:
            if run_id in self._live:
                return {"answer": None, "state": self._live[run_id].to_dict()}
        p = self.runs_dir / f"{run_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def list(self, limit=50) -> list[dict]:
        out = []
        for p in sorted(self.runs_dir.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True)[:limit]:
            try:
                a = json.loads(p.read_text(encoding="utf-8"))["answer"]
                out.append({"run_id": a["run_id"], "asof": a["asof"], "question": a["question"],
                            "regime": a["forecast"]["most_likely"], "warning": a["scenario"]["warning_level"],
                            "status": a["status"], "review": a["human_review"]["required"]})
            except Exception:  # noqa: BLE001
                continue
        return out
