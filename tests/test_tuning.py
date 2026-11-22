import json

import pandas as pd

from volsentinel.agents.analyst import ScenarioAnalyst
from volsentinel.config import Settings
from volsentinel.state import RunState
from volsentinel.tune import apply_gate, held_out_check, score, search


def _frame():
    # 6 weeks: two stressed weeks, four quiet ones; all start with a raw 'warning' except one
    return pd.DataFrame({
        "outcome": ["Stressed", "Calm", "Calm", "Stressed", "Elevated", "Calm"],
        "tot_warning_raw": ["severe", "warning", "warning", "none", "severe", "watch"],
        "tot_g_score": [0.7, 0.5, 0.6, 0.4, 0.55, 0.5],
        "tot_g_support": [0.75, 0.25, 0.5, 0.0, 0.5, 0.3],
        "tot_g_p_stress": [0.3, 0.1, 0.12, 0.2, 0.05, 0.1],
        "tot_g_precedent": [True, False, False, False, False, False],
        "tot_g_risk": [8, 2, 5, 0, None, 1],
    })


def test_gate_lowers_only_failing_warnings():
    df = _frame()
    p = {"warn_min_score": 0.0, "warn_min_support": 0.0, "warn_min_p_stress": 0.25, "warn_min_risk_headlines": 6}
    g = apply_gate(df, p)
    assert list(g) == ["severe", "watch", "watch", "none", "watch", "watch"]
    # no headline text (None) skips only the headline check: p_stress still fails for week 5
    assert score(df, g)["false_alarms"] == 0 and score(df, g)["hits"] == 1


def test_search_prefers_lower_cost_then_simpler():
    best, results = search(_frame())
    assert results[0]["cost"] <= results[-1]["cost"]
    assert score(_frame(), apply_gate(_frame(), best))["hits"] == 1


def test_held_out_rule_rejects_a_gate_that_loses_a_hit():
    df = _frame()
    strict = {"warn_min_score": 0.9, "warn_min_support": 0.0, "warn_min_p_stress": 0.0, "warn_min_risk_headlines": 0}
    res = held_out_check(df, strict)
    assert not res["adopted"] and res["after"]["hits"] < res["before"]["hits"]


def test_unadopted_tuning_is_not_loaded(tmp_path, monkeypatch):
    f = tmp_path / "tuned.json"
    f.write_text(json.dumps({"params": {"warn_min_p_stress": 0.25}, "adopted": False}))
    monkeypatch.setenv("VOLSENTINEL_TUNED", str(f))
    s = Settings()
    assert s.warn_min_p_stress == 0.0 and "not adopted" in s.tuned_source
    f.write_text(json.dumps({"params": {"warn_min_p_stress": 0.25}, "adopted": True, "fitted_on": "val"}))
    s = Settings()
    assert s.warn_min_p_stress == 0.25 and "val" in s.tuned_source


def test_analyst_gate_matches_offline_gate():
    s = Settings()
    s.warn_min_p_stress, s.warn_min_risk_headlines = 0.25, 6
    st = RunState("q", "2024-01-01", "replay")
    st.add_evidence("news_search", "news_risk_n", 3, "Risk headlines", "3")
    an = ScenarioAnalyst(st, None, s)
    node = {"id": "d3-x", "family": "insolvency_contagion", "predicted": [{}, {}], "refuting": [],
            "confirmed_n": 2}
    g = an.gate(node, {"score": 0.8}, {"probs": [0.3, 0.4, 0.3]}, "severe")
    assert g["level"] == "watch" and any("risk headlines" in x for x in g["failed"])
    st.add_evidence("news_search", "news_risk_n", 7, "Risk headlines", "7")
    assert an.gate(node, {"score": 0.8}, {"probs": [0.3, 0.4, 0.3]}, "severe")["level"] == "severe"
