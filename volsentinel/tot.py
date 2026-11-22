"""Tree-of-Thought controller: beam search over scenarios (Checkpoint 4.1).

Depth 0  as-of state (HAR forecast, signals, headlines, precedents)
Depth 1  scenario family (5)          -> keep the best 2
Depth 2  specific mechanism (2 each)  -> keep the best 2
Depth 3  conclusion (1 each)          -> best leaf is the answer, runner-up the alternative
Budget: 11 node evaluations (5 + 4 + 2), 14 with the protected tail slot.

Protected tail slot: the most severe scenario that has a gate-passing precedent or
at least one confirmed signal keeps a reserved place through depth 2. It can be
pruned only when evidence actively refutes it, not merely for lack of confirmation.
"""
from __future__ import annotations

import functools
import time

from .scenarios import FAMILY


class Controller:
    NAME = "tot_controller"

    def __init__(self, st, analyst, critic, settings, har: dict, deadline: float):
        self.st, self.an, self.cr, self.s, self.har = st, analyst, critic, settings, har
        self.deadline = deadline
        self.evals = 0
        self.tail_evals = 0
        self.nodes: dict[str, dict] = {}
        self.budget_exhausted = False

    # ---------------------------------------------------------------- helpers
    def _cmp(self, a, b):
        if abs(a["score"] - b["score"]) > self.s.tie_margin:
            return -1 if a["score"] > b["score"] else 1
        if a["confirmed_n"] != b["confirmed_n"]:                       # tie: more confirmed signals
            return -1 if a["confirmed_n"] > b["confirmed_n"] else 1
        ca, cb = a["components"].get("consistency", 0), b["components"].get("consistency", 0)
        if ca != cb:                                                   # then closer to HAR
            return -1 if ca > cb else 1
        return -1 if a["score"] >= b["score"] else 1

    def rank(self, nodes):
        return sorted(nodes, key=functools.cmp_to_key(self._cmp))

    def _can_eval(self, extra_allowed: bool) -> bool:
        over = (self.evals >= self.s.max_node_evals_with_tail if extra_allowed
                else self.evals - self.tail_evals >= self.s.max_node_evals)
        if over or time.time() > self.deadline:
            self.budget_exhausted = True
            return False
        return True

    def _eval(self, node, tail=False) -> dict | None:
        if not self._can_eval(tail):
            node["status"] = "skipped (budget)"
            self.nodes[node["id"]] = node
            return None
        self.cr.score(node, self.har["modal"])
        self.evals += 1
        self.tail_evals += int(tail)
        node["eval_order"] = self.evals
        self.nodes[node["id"]] = node
        return node

    def _protectable(self, n) -> bool:
        fam = FAMILY[n["family"]]
        prec_ok = any(s["state"] == "confirmed" for s in n["predicted"] if s["name"] == fam.precedent_signal)
        return not n["refuted"] and (prec_ok or n["confirmed_n"] >= 1)

    def _pick_tail(self, candidates, beam):
        top_sev = max((b["severity"] for b in beam), default=-1)
        pool = [c for c in candidates if c["severity"] > top_sev and self._protectable(c)]
        if not pool:
            return None
        t = sorted(pool, key=lambda c: (-c["severity"], -(c["score"] or 0)))[0]
        t["status"] = "tail"
        self.st.record(self.NAME, "note", f"protected tail slot -> {t['id']} (severity {t['severity']}, "
                                          f"{t['confirmed_n']} confirmed signal(s), not refuted)")
        return t

    def _select(self, scored, level):
        alive = [n for n in scored if n["score"] >= self.s.prune_below]
        for n in scored:
            if n not in alive:
                n["status"] = "pruned"
        beam = self.rank(alive)[: self.s.beam_width]
        for n in beam:
            n["status"] = "kept"
        for n in alive:
            if n not in beam:
                n["status"] = "pruned"
        self.st.record(self.NAME, "note", f"depth {level} beam: " + ", ".join(f"{n['id']} {n['score']:.2f}" for n in beam)
                       + (f"; pruned {sum(n['status'] == 'pruned' for n in scored)}" if scored else ""))
        return beam

    # ---------------------------------------------------------------- search
    def run(self) -> dict:
        self.st.record(self.NAME, "plan", f"beam search width {self.s.beam_width}, max depth 3, "
                                          f"{self.s.max_node_evals} evaluations ({self.s.max_node_evals_with_tail} with tail)")
        root = {"id": "root", "parent": None, "depth": 0, "label": "As-of state", "status": "kept", "score": None,
                "thought": f"HAR {self.har['modal']} band; signals, headlines and precedents gathered", "family": None}
        self.nodes["root"] = root
        d1 = [n for n in (self._eval(n) for n in self.an.propose_scenarios()) if n]
        beam1 = self._select(d1, 1)
        tail1 = self._pick_tail([n for n in d1 if n not in beam1], beam1)

        d2, d2_tail = [], []
        for parent in beam1:
            d2 += [n for n in (self._eval(c) for c in self.an.propose_mechanisms(parent, self.s.depth2_children)) if n]
        if tail1:
            d2_tail = [n for n in (self._eval(c, tail=True) for c in self.an.propose_mechanisms(tail1, self.s.depth2_children)) if n]
        beam2 = self._select(d2, 2)
        tail2 = None
        if tail1 and d2_tail:
            cands = [c for c in self.rank(d2_tail) if not c["refuted"]]
            for c in d2_tail:
                c["status"] = "pruned"
            if cands and (self._protectable(cands[0]) or cands[0]["score"] >= self.s.prune_below):
                tail2 = cands[0]
                tail2["status"] = "tail"
            else:
                self.st.record(self.NAME, "note", f"tail {tail1['id']} released: its mechanisms were refuted "
                                                  "or had no confirming evidence and scored below the prune line")

        leaves = []
        for i, parent in enumerate(beam2):
            other = beam2[1 - i] if len(beam2) > 1 else tail2
            leaf = self.an.conclude(parent, self.har, other)
            if self._eval(leaf):
                leaves.append(leaf)
        if tail2:
            leaf = self.an.conclude(tail2, self.har, beam2[0] if beam2 else None)
            if self._eval(leaf, tail=True):
                leaf["tail"] = True
                leaves.append(leaf)

        valid = [l for l in leaves if l["score"] >= self.s.prune_below and l.get("citation", {}).get("ok")]
        for l in leaves:
            l["status"] = "pruned" if l not in valid else "leaf"
        ranked = self.rank(valid)
        best = ranked[0] if ranked else None
        runner = ranked[1] if len(ranked) > 1 else None
        dominant = bool(best and best["score"] >= self.s.accept_score and
                        (runner is None or best["score"] - runner["score"] >= self.s.accept_margin))
        co_leading = bool(best and runner and abs(best["score"] - runner["score"]) <= self.s.tie_margin
                          and best["confirmed_n"] == runner["confirmed_n"]
                          and best["components"]["consistency"] == runner["components"]["consistency"])
        if best:
            best["status"] = "selected"
            if runner:
                runner["status"] = "alternative"
        self.st.tree = self.nodes
        self.st.record(self.NAME, "note",
                       (f"selected {best['id']} ({best['score']:.2f})" + (f", runner-up {runner['id']} ({runner['score']:.2f})" if runner else "")
                        + ("; dominant" if dominant else "; not dominant") + ("; co-leading" if co_leading else ""))
                       if best else "every branch pruned: no dominant scenario")
        return {"best": best, "runner_up": runner, "dominant": dominant, "co_leading": co_leading,
                "evals": self.evals, "budget_exhausted": self.budget_exhausted, "tail_used": bool(tail1)}


CATEGORY_TO_FAMILY = {"insolvency": "insolvency_contagion", "hack": "insolvency_contagion",
                      "depeg": "insolvency_contagion", "liquidation": "leverage_flush", "macro": "macro_risk_off",
                      "scheduled": "scheduled_event", "regulation": "scheduled_event"}


def single_chain(st, analyst, critic, har: dict, first_category: str | None, funding_extreme: bool) -> dict:
    """Linear chain-of-thought baseline for evaluation: commit to the scenario suggested
    by the first vivid headline (or funding, or quiet drift) and conclude without
    comparing alternatives."""
    fam = CATEGORY_TO_FAMILY.get(first_category) or ("leverage_flush" if funding_extreme else "quiet_drift")
    d1 = next(n for n in analyst.propose_scenarios() if n["family"] == fam)
    critic.score(d1, har["modal"])
    d2 = analyst.propose_mechanisms(d1, 1)[0]
    critic.score(d2, har["modal"])
    leaf = analyst.conclude(d2, har, None)
    critic.score(leaf, har["modal"])
    return leaf
