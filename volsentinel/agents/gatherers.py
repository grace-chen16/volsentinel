"""Market agent (agent 2) and News & retrieval agent (agent 3).

Both run a ReAct loop: write a short reasoning step that says what is known, what
is missing and which tool fills the gap; take one action; read the observation;
repeat. Observations change the plan (a surprising value opens a follow-up query,
an error triggers a backup source) and stopping is explicit.
"""
from __future__ import annotations

from ..tools import Observation, ToolRegistry


class ReActAgent:
    NAME = "agent"
    MAX_STEPS = 8

    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        self.st = tools.state
        self.steps = 0
        self.obs: dict[str, Observation] = {}
        self.gaps: list[str] = []

    def reason(self, text: str):
        self.st.record(self.NAME, "reason", text)

    def act(self, tool: str, why: str, **args) -> Observation:
        self.steps += 1
        self.reason(why)
        o = self.tools.call(self.NAME, tool, **args)
        self.obs[tool + (f":{args['feed']}" if "feed" in args else "") + (f":{args['signal']}" if "signal" in args else "")] = o
        return o

    def can_continue(self) -> bool:
        return self.steps < self.MAX_STEPS and self.tools.remaining() > 2 and self.tools.time_left() > 5


class MarketAgent(ReActAgent):
    NAME = "market_agent"

    def run(self) -> dict:
        har = self.act("har_forecast", "I need a baseline before anything else: run the HAR forecast on realized volatility.")
        if not har.ok:
            self.gaps.append("HAR forecast failed")
            self.reason("Without HAR there is no anchor; report the failure to the orchestrator.")
            return self.result(None)
        independent = 0
        fund = self.act("funding_rates", f"HAR says {har.data['modal']}. What is driving it? Check leverage through "
                                         "funding rates, an independent signal.")
        if fund.ok:
            independent += 1
            z = fund.data.get("funding_z")
            if z is not None and abs(z) > 2 and self.can_continue():
                sig = "funding_spike" if z > 0 else "funding_crash"
                self.act("history_query", "Funding is far outside its recent range, a sign of crowded positioning. "
                                          "Has this happened before, and what followed?", signal=sig)
        else:
            self.gaps.append("funding rates unavailable")
            self.reason("Funding failed; leverage stays unverified and any branch relying on it is capped.")
        if self.can_continue():
            vx = self.act("vix_fear", "Check cross-asset stress (VIX) and crypto sentiment (Fear & Greed).")
            if vx.ok:
                independent += 1
                if (vx.data.get("vix_chg5") or 0) > 0.20 and self.can_continue():
                    self.act("history_query", "VIX jumped sharply; check what followed past equity-vol shocks.", signal="vix_shock")
            else:
                self.gaps.append("VIX / Fear & Greed unavailable")
        if self.can_continue():
            snap = self.act("market_snapshot", "Look at price action: drawdown and the size of the latest daily move.")
            if snap.ok:
                d = snap.data
                if d.get("rv1") and d.get("rv7") and d["rv1"] > 2.5 * d["rv7"] and self.can_continue():
                    self.act("history_query", "The last daily move dwarfs the past week's; check past volatility jumps.",
                             signal="vol_jump")
                elif d.get("ret7") is not None and d["ret7"] < -0.10 and self.can_continue():
                    self.act("history_query", "A sharp weekly drawdown; check what followed past drawdowns.", signal="drawdown")
        if self.can_continue():
            peg = self.act("stablecoin_peg", "Check stablecoin pegs; a depeg is an early contagion signal.")
            if not peg.ok:
                self.gaps.append("stablecoin prices unavailable")
        self.reason(f"Stop: HAR checked against {independent} independent signal(s)"
                    + (f"; gaps: {', '.join(self.gaps)}" if self.gaps else "") + ".")
        return self.result(har.data, independent)

    def result(self, har, independent=0):
        return {"har": har, "independent_checks": independent, "gaps": self.gaps, "steps": self.steps}


class NewsAgent(ReActAgent):
    NAME = "news_agent"

    def run(self) -> dict:
        news = self.act("news_search", "I need this week's event headlines: search the primary news feed for the last three days.",
                        feed="primary", days=3)
        if not news.ok:
            self.reason(f"The primary feed failed ({news.error}). Try the backup source.")
            news = self.act("news_search", "Query the backup feed.", feed="backup", days=3)
            if not news.ok:
                self.gaps.append("both news feeds failed")
        queries = news.data.get("risk_headlines", []) if news.ok else []
        if news.ok and news.data.get("flagged_injection"):
            self.reason(f"{len(news.data['flagged_injection'])} headline(s) contain instruction-like text; they stay "
                        "quarantined as data and are not used as queries.")
        precedents = None
        if queries and self.can_continue():
            precedents = self.act("retrieve_precedents", f"{len(queries)} risk headline(s) found. Has something like this "
                                                         "happened before? Search the event cards with the similarity gate.",
                                  queries=queries)
        elif not queries:
            self.reason("No risk headlines to search with, so no precedent search; the trace will say so.")
            self.tools.state.add_evidence("retrieve_precedents", "precedents_n", 0,
                                          "Precedents passing the retrieval gate (no risk headlines to search)", "0",
                                          agent=self.NAME)
        mem = None
        if self.can_continue():
            mem = self.act("memory_search", "Check my own record: similar past setups and whether I over- or under-called them.")
        self.reason("Stop: headlines, precedents and my own track record are in the shared state.")
        return {"news": news.data if news.ok else None, "queries": queries,
                "precedents": precedents.data if precedents and precedents.ok else {"precedents": [], "rejected": []},
                "memory": mem.data if mem and mem.ok else None, "gaps": self.gaps, "steps": self.steps}
