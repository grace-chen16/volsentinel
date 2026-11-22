"""Orchestrator (agent 1): plans each run, starts the other agents, enforces time
and tool budgets, and falls back to a HAR-only answer if a step fails.

Run order (Checkpoint 5.1):
  1. plan the run and validate inputs;
  2. market and news agents gather data in parallel (graph-based, one-way hand-offs);
  3. scenario analyst and risk critic iterate (Tree of Thought, then at most two
     validation rounds; the critic may ask the orchestrator to re-fetch missing data);
  4. only critic-approved output becomes the answer.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..config import DISCLAIMER
from ..data.market import REGIMES
from ..safety import (ADVICE_DECLINE, is_advice_request, outside_two_year_range, sanitize_text,
                      validate_market)
from ..scenarios import FAMILY, WARN_LEVELS
from ..tools import RunInputs, ToolRegistry, pct
from ..tot import Controller
from .analyst import ScenarioAnalyst
from .critic import RiskCritic
from .gatherers import MarketAgent, NewsAgent

log = logging.getLogger(__name__)
NAME = "orchestrator"
REFETCHABLE = {"funding_rates": "funding_rates", "vix_fear": "vix_fear", "stablecoin_peg": "stablecoin_peg",
               "market_snapshot": "market_snapshot"}


class RunError(ValueError):
    pass


def resolve_asof(ctx, mode: str, asof: str | None):
    if mode == "live":
        try:
            mkt, status = ctx.live_market()
            live_ok = True
        except Exception as e:  # noqa: BLE001
            mkt, status, live_ok = ctx.mkt, {"live": f"error: {type(e).__name__}: {e}"[:200]}, False
        d = mkt.dropna(subset=["rv7", "thr_calm"]).index.max()
        return mkt, d, status, live_ok
    mkt = ctx.mkt
    if asof:
        try:
            d = pd.Timestamp(asof).normalize()
        except Exception as e:  # noqa: BLE001
            raise RunError(f"invalid as_of date {asof!r}") from e
        ok = mkt.dropna(subset=["rv7", "thr_calm", "rv30"])
        if d not in ok.index:
            raise RunError(f"as_of {d.date()} outside the available market history "
                           f"({ok.index.min().date()} to {ok.index.max().date()})")
    else:
        d = ctx.default_asof
    return mkt, d, {}, True


def run_forecast(ctx, question: str, asof: str | None = None, mode: str = "replay",
                 headlines: list[str] | None = None, simulate_failures: list[str] | None = None,
                 use_llm: bool = True, trigger: str = "interactive", save: bool = True,
                 baseline: bool = False) -> dict:
    s = ctx.settings
    question_clean, q_flag = sanitize_text(question or "Is a volatility spike likely this week?", 500)
    mkt, d, live_status, live_ok = resolve_asof(ctx, mode, asof)
    st = ctx.store.new_run(question_clean, str(d.date()), mode)
    deadline = st.started + s.time_limit_s
    st.record(NAME, "plan", f"As-of {d.date()} ({mode}). Plan: validate inputs -> market and news agents in "
                            f"parallel -> Tree of Thought -> critic approval. Budgets: {s.max_tool_calls} tool calls, "
                            f"{s.time_limit_s:.0f}s, {s.max_critic_rounds} critic rounds.")
    advice = is_advice_request(question_clean)
    if advice:
        st.record(NAME, "note", "The question asks for personal trading advice: that part is declined and the run "
                                "is flagged for review; the volatility forecast is still provided.")
    if q_flag:
        st.record(NAME, "note", "The question contains instruction-like text; it is treated as data only.")

    # ---- input guardrails ---------------------------------------------------
    reference = (pd.Timestamp.utcnow().tz_localize(None) if mode == "live" else d + pd.Timedelta(days=1))
    issues = validate_market(mkt, d, reference, s.max_data_age_hours)
    if mode == "live" and not live_ok:
        issues.append({"feed": "live", "issue": "live feeds unavailable; fell back to archived data", "severity": "high"})
    for k, v in live_status.items():
        if v != "ok":
            issues.append({"feed": f"live:{k}", "issue": v, "severity": "medium"})
    out_of_range = outside_two_year_range(mkt, d)
    for i in issues:
        st.record(NAME, "note", f"input check: {i['feed']} {i['issue']}")

    # user-supplied headlines are data; the news tool sanitizes them and quarantines injections
    user_headlines = [str(h)[:500] for h in (headlines or [])[:50] if str(h).strip()]
    ri = RunInputs(ctx=ctx, mkt=mkt, asof=d, mode=mode, user_headlines=user_headlines,
                   simulate_failures=set(simulate_failures or []))
    tools = ToolRegistry(ri, st, s.max_tool_calls, s.time_limit_s)

    # ---- parallel gathering ---------------------------------------------------
    st.record(NAME, "handoff", "start market_agent and news_agent in parallel")
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gather") as ex:
        fm = ex.submit(MarketAgent(tools).run)
        fn = ex.submit(NewsAgent(tools).run)
        try:
            market = fm.result(timeout=max(1.0, deadline - time.time()))
        except Exception as e:  # noqa: BLE001
            market = {"har": None, "gaps": [f"market agent failed: {type(e).__name__}: {e}"], "independent_checks": 0}
        try:
            news = fn.result(timeout=max(1.0, deadline - time.time()))
        except Exception as e:  # noqa: BLE001
            news = {"news": None, "queries": [], "precedents": {"precedents": [], "rejected": []}, "memory": None,
                    "gaps": [f"news agent failed: {type(e).__name__}: {e}"]}
    har = market.get("har")
    if har is None:
        st.record(NAME, "error", "HAR forecast unavailable: no valid answer can be produced.")
        ans = _failed_answer(st, d, mode, question_clean, market, news, issues)
        if save:
            ctx.store.save(st, ans)
        return ans

    llm = ctx.llm if use_llm else None
    analyst = ScenarioAnalyst(st, llm, s)
    critic = RiskCritic(st, s, llm)

    # ---- critic: re-fetch missing data once -------------------------------------
    failed = [e.tool for e in st.log if e.kind == "error" and e.tool in REFETCHABLE and e.agent == "market_agent"]
    for tname in dict.fromkeys(failed):
        if tools.remaining() <= 0 or time.time() > deadline:
            break
        st.send("risk_critic", NAME, "refetch", {"tool": tname, "reason": "input missing for scenario checks"})
        o = tools.call(NAME, tname)
        st.record(NAME, "note", f"re-fetch {tname}: {'ok' if o.ok else 'still failing - branches depending on it stay unverified'}")

    # ---- Tree of Thought + critic rounds --------------------------------------
    status, tot, leaf, chk, rounds = "approved", None, None, None, 0
    fallback_reason = None
    try:
        tot = Controller(st, analyst, critic, s, har, deadline).run()
        leaf = tot["best"]
        if leaf is not None:
            for rounds in range(1, s.max_critic_rounds + 1):
                chk = critic.review_final(leaf)
                if chk["ok"]:
                    break
                if rounds < s.max_critic_rounds:
                    st.record("scenario_analyst", "reason", "Critic rejected the draft; rewrite it from the cited "
                                                            "evidence only.")
                    leaf["explanation"] = analyst.draft(leaf, har, tot["runner_up"], feedback=chk["issues"])
            if not chk["ok"]:
                fallback_reason = "critic did not approve the explanation within two rounds"
                leaf = None
    except Exception as e:  # noqa: BLE001
        log.exception("tree of thought failed")
        fallback_reason = f"scenario analysis failed ({type(e).__name__}: {e})"
        st.record(NAME, "error", fallback_reason)
        tot, leaf = tot or {"best": None, "runner_up": None, "dominant": False, "co_leading": False,
                            "evals": 0, "budget_exhausted": False}, None

    if fallback_reason:
        status = "har_only_fallback"
    elif leaf is None:
        status = "no_dominant_scenario"
    elif not tot["dominant"] or tot["budget_exhausted"]:
        status = "approved_low_confidence"

    if leaf is None:
        text = har_only_text(st, fallback_reason or "every scenario branch was pruned: no dominant scenario")
        chk = critic.check_text(text)
        st.record("risk_critic", "note", f"HAR-only answer checked: {'ok' if chk['ok'] else chk['issues']}")
    else:
        text = leaf["explanation"]

    # ---- human review triggers -------------------------------------------------
    p = np.array(har["probs"])
    warning = leaf["warning"] if leaf else ("watch" if har["modal"] == "Stressed" else "none")
    reasons = []
    if leaf and ((har["modal"] == "Calm" and WARN_LEVELS.index(warning) >= 2)
                 or (har["modal"] == "Stressed" and leaf["family"] == "quiet_drift")):
        reasons.append("high-impact disagreement: HAR and the leading scenario point in different directions")
    top2 = np.sort(p)[-2:]
    if leaf is None or leaf["score"] < s.accept_score:
        reasons.append("low confidence: no scenario scored at least 0.70")
    if top2[1] - top2[0] < s.regime_close_margin:
        reasons.append("low confidence: the top two regimes are within 10 points")
    high = [i for i in issues if i["severity"] == "high"]
    gaps = (market.get("gaps") or []) + (news.get("gaps") or [])
    if high or gaps or out_of_range:
        reasons.append("data integrity: " + "; ".join([f"{i['feed']} {i['issue']}" for i in high] + gaps + out_of_range))
    firms = (chk or {}).get("firms_named_insolvent") or []
    if firms:
        reasons.append(f"policy: output names {', '.join(firms)} as insolvent")
    if advice:
        reasons.append("policy: user asked for personal trading advice (declined and referred)")
    drift = ctx.memory.drift(s.drift_window_weeks)
    if drift.get("drift"):
        reasons.append("drift: calibration or critic rejection rate worsened over the last eight scored weeks; "
                       "automated daily brief paused")
    required = bool(reasons)
    release = ("hold for human review" if required and trigger == "scheduled"
               else "release with uncertainty label" if required else "release")
    confidence = ("high" if status == "approved" and not required else
                  "low" if status in ("har_only_fallback", "no_dominant_scenario") or len(reasons) >= 2 else "medium")

    runner = tot["runner_up"] if tot else None
    prec = news.get("precedents") or {"precedents": [], "rejected": []}
    answer = {
        "run_id": st.run_id, "asof": str(d.date()), "mode": mode, "trigger": trigger, "question": question_clean,
        "status": status,
        "headline": headline(har, leaf, warning, status),
        "forecast": {"probabilities": dict(zip(REGIMES, map(float, p))), "most_likely": har["modal"],
                     "point_rv7": har["point_rv7"], "rv7_now": har["rv7_now"], "regime_now": har["regime_now"],
                     "thresholds": {"calm_below": har["thr_calm"], "stressed_above": har["thr_stress"]},
                     "naive_probabilities": har["naive_probs"],
                     "method": f"HAR on log realized vol (blend weight {s.blend_weight}: precedents explain, they do not move the numbers)"},
        "scenario": ({"name": leaf["label"], "family": leaf["family"], "family_label": FAMILY[leaf["family"]].label,
                      "mechanism": leaf["mechanism"], "warning_level": warning, "score": leaf["score"],
                      "warning_gate": leaf.get("warning_gate"),
                      "components": leaf["components"], "dominant": tot["dominant"], "co_leading": tot["co_leading"],
                      "confirmed_signals": [x for x in leaf["predicted"] if x["state"] == "confirmed"],
                      "unconfirmed_signals": [x for x in leaf["predicted"] if x["state"] != "confirmed"],
                      "refuted_by": [x["text"] for x in leaf["refuting"]]}
                     if leaf else {"name": "No dominant scenario", "family": None, "warning_level": warning,
                                   "score": None, "dominant": False}),
        "alternative": ({"name": runner["label"], "family": runner["family"], "score": runner["score"],
                         "warning_level": runner.get("warning")} if runner and leaf else None),
        "explanation": (ADVICE_DECLINE + " " if advice else "") + text,
        "advice_declined": advice,
        "precedents": prec.get("precedents", []), "rejected_precedents": prec.get("rejected", []),
        "news": {"queries": news.get("queries", []),
                 "source": (news.get("news") or {}).get("source"),
                 "n_headlines": (news.get("news") or {}).get("n_headlines"),
                 "categories": {k: len(v) for k, v in ((news.get("news") or {}).get("categories") or {}).items()},
                 "quarantined": (news.get("news") or {}).get("flagged_injection", []),
                 "recent": (news.get("news") or {}).get("recent", [])},
        "memory": news.get("memory"),
        "human_review": {"required": required, "reasons": reasons, "release": release},
        "confidence": confidence,
        "data_quality": {"issues": issues, "gaps": gaps, "outside_two_year_range": out_of_range},
        "critic": {"rounds": rounds, "rejections": critic.rejections, "notes": critic.notes,
                   "citation_check": chk},
        "metrics": {"latency_s": round(st.elapsed(), 2), "tool_calls": tools.calls,
                    "node_evals": tot["evals"] if tot else 0, "budget_exhausted": bool(tot and tot["budget_exhausted"]),
                    "grounded_share": (chk or {}).get("grounded_share"), "llm": bool(llm and llm.available)},
        "fingerprint": ri.fingerprint(),
        "evidence": {k: {"label": v.label, "display": v.display, "tool": v.tool, "agent": v.agent}
                     for k, v in st.evidence.items()},
        "disclaimer": DISCLAIMER,
    }
    if baseline:
        # evaluation only: the linear chain-of-thought baseline on the same evidence
        from ..tot import single_chain
        risk = (news.get("news") or {}).get("risk_headlines") or []
        from ..data.news import categorize
        first_cat = next((c for h in risk for c in categorize(h)), None)
        fz = st.find_evidence("funding_z")
        b = single_chain(st, ScenarioAnalyst(st, None, s), RiskCritic(st, s, None), har, first_cat,
                         bool(fz and fz.value is not None and abs(fz.value) > 2))
        answer["baseline_single_chain"] = {"family": b["family"], "mechanism": b["mechanism"],
                                           "warning_level": b["warning"], "score": b["score"],
                                           "warning_gate": b.get("warning_gate"),
                                           "citation_ok": b.get("citation", {}).get("ok")}
    st.meta["summary"] = answer["headline"]
    st.record(NAME, "note", f"answer: {answer['headline']} [{status}; review {'required' if required else 'not required'}]")
    if save:
        ctx.store.save(st, answer)
        if ctx.memory.save(answer):
            st.record(NAME, "note", "forecast written to long-term memory; it will be scored after seven days")
        ctx.memory.score_due(mkt)
    answer["tree"] = st.tree
    answer["trace"] = [e.__dict__ for e in st.log]
    return answer


def headline(har, leaf, warning, status) -> str:
    p = dict(zip(REGIMES, har["probs"]))
    base = f"{har['modal']} most likely next week (Calm {pct(p['Calm'])}, Elevated {pct(p['Elevated'])}, Stressed {pct(p['Stressed'])})"
    if leaf:
        return f"{base}; leading scenario: {FAMILY[leaf['family']].label.lower()}, warning level {warning}"
    if status == "har_only_fallback":
        return f"{base}; HAR-only fallback"
    return f"{base}; no dominant scenario"


def har_only_text(st, reason: str) -> str:
    g = lambda k: st.find_evidence(k)
    pc, pe, ps, hp, tc, ts = (g(k) for k in ("p_calm", "p_elevated", "p_stressed", "har_point", "thr_calm", "thr_stress"))
    return (f"HAR-only forecast: Calm {pc.display}, Elevated {pe.display}, Stressed {ps.display} [{pc.id}, {pe.id}, {ps.id}]. "
            f"The point forecast is {hp.display} realized volatility against a Calm threshold of {tc.display} and a "
            f"Stressed threshold of {ts.display} [{hp.id}, {tc.id}, {ts.id}]. "
            f"No scenario explanation is attached because {reason}.")


def _failed_answer(st, d, mode, question, market, news, issues) -> dict:
    return {"run_id": st.run_id, "asof": str(d.date()), "mode": mode, "question": question, "status": "failed",
            "headline": "No forecast: the HAR baseline could not be computed",
            "forecast": None, "scenario": {"name": None, "warning_level": None}, "alternative": None,
            "explanation": "The HAR forecast failed, so VolSentinel cannot give a grounded answer for this date.",
            "human_review": {"required": True, "reasons": ["data integrity: HAR baseline unavailable"] + market.get("gaps", []),
                             "release": "hold for human review"},
            "data_quality": {"issues": issues, "gaps": market.get("gaps", []) + news.get("gaps", [])},
            "critic": {"rounds": 0, "rejections": 0, "notes": []},
            "metrics": {"latency_s": round(st.elapsed(), 2), "tool_calls": None, "node_evals": 0},
            "disclaimer": DISCLAIMER, "trace": [e.__dict__ for e in st.log]}
