# VolSentinel

A read-only, multi-agent forecaster of next-week Bitcoin volatility risk. For a question such
as *"Is a volatility spike likely this week?"*, it does four things:

- classifies the next seven days as **Calm**, **Elevated** or **Stressed**, with probabilities;
- names the scenario that best explains the risk and gives it a warning level;
- writes an explanation in which every number is cited to a tool output;
- says when a person should review the answer before it goes out.

It informs decisions. It never trades.

This is the implementation of the capstone proposals in `../proposal/`:

| Checkpoint | Design element | Where it lives |
|---|---|---|
| 2 Reasoning, memory, tools | ReAct loop, short-term log, long-term knowledge and experience memory, read-only tools | `agents/gatherers.py`, `state.py`, `memory.py`, `tools.py` |
| 3.1 RAG and retrieval | Risk-headline queries, event cards, top-3 consensus score, semantic >= 0.20 and condition >= 0.30 gate, 30-day time guard, visible rejections, blend weight 0 | `data/cards.py`, `data/news.py`, `tools.py: retrieve_precedents` |
| 4.1 Tree of Thought | 5 scenario families, 2 mechanisms each, conclusions at depth 3; beam width 2; weighted rubric; prune < 0.40; unverified cap 0.50; protected tail slot; 11 (14) evaluations | `scenarios.py`, `tot.py`, `agents/analyst.py`, `agents/critic.py` |
| 5.1 Multi-agent | Orchestrator, Market agent, News & retrieval agent, Scenario analyst, Risk critic; parallel gathering, two critic rounds at most, re-fetch requests, shared state, HAR-only fallback | `agents/orchestrator.py`, `state.py`, `mcp_server.py` |
| 6.1 Safety and evaluation | Freshness and range checks, injection quarantine, 15 tool calls / 90 s budget, citation check, no trade language, disclaimer, five human-review triggers, evaluation metrics, drift pause | `safety.py`, `agents/critic.py`, `agents/orchestrator.py`, `evaluate.py`, `memory.py` |

## How a run works

```
question / schedule
      |
 1 Orchestrator -- plans, validates inputs, enforces budgets, HAR-only fallback
      |                      (parallel, one-way hand-offs)
      +--> 2 Market agent (ReAct): HAR -> funding -> [history] -> VIX/F&G -> price -> stablecoin peg
      +--> 3 News & retrieval agent (ReAct): news (primary -> backup) -> gated precedent search -> own record
      |
 4 Scenario analyst  <-- Tree of Thought, beam 2 -->  5 Risk critic (scores 30% judgement + 70% heuristics)
      |                                                     | citation check, at most 2 rounds,
      +------------- critic-approved answer <---------------+ re-fetch requests to the orchestrator
      |
 answer + reasoning trace -> long-term memory (scored after 7 days)
```

All agents read and write one shared state (`state.py`): every tool number becomes evidence
`E1, E2, ...`, every step goes in the log, and the tree is stored with scores. The critic checks
that each number in the explanation appears in an evidence item cited in the same sentence.
**HAR alone sets the probabilities.** Precedents and scenarios explain the forecast and set the
warning level. They move the numbers only once a blend passes a walk-forward test (blend weight 0).

## Data

Everything is read-only and needs no API keys.

| Source | Used for | Location |
|---|---|---|
| Project database | BTC and stablecoin prices, funding, VIX, Fear & Greed, news-flow counts (2017 to Nov 2025) | `../crypto-vol-api/backend/db/crypto_data.db` |
| Event cards (22 episodes) | Precedent knowledge base | `volsentinel/data/event_cards.json` |
| CryptoPanic archive (111k headlines, 2019-2025) | Primary news in replay and precedent evidence | `../../PredictiveModel/RawData/CryptoNews/news_crypto.csv` |
| Coinbase, OKX, Yahoo, alternative.me, Cointelegraph and CoinDesk RSS | Live mode | public endpoints (`data/live.py`) |

**Data is not included in this repository.** The project database and the news archive come from
the author's earlier course project. To run VolSentinel elsewhere:

- point `VOLSENTINEL_DB` at a SQLite file with the same `crypto_px` and `master_dataset_pred` tables;
- point `VOLSENTINEL_NEWS_CSV` at the public CryptoPanic dataset
  (https://github.com/soheilrahsaz/cryptoNewsDataset).

Without the news archive the system still runs. It falls back to the backup news source and fits
the card embeddings on the cards alone. Tests that need the database are skipped when it is
missing. The `results/` folder holds every evaluation output, so you can review the results
without the data.

Paths can be overridden with `VOLSENTINEL_DB`, `VOLSENTINEL_NEWS_CSV`, `VOLSENTINEL_CARDS`,
`VOLSENTINEL_RUNS` and `VOLSENTINEL_MEMORY`. The first run builds the news index (about 45 s).
The index is cached in `.cache/`.

## Run it

```powershell
cd "volsentinel"
pip install -r requirements.txt
python -m volsentinel serve --port 8000          # web UI at http://127.0.0.1:8000/
python -m volsentinel ask --as-of 2022-11-07 --trace
python -m volsentinel ask --live                 # today, from public feeds
python -m volsentinel ask "Is a spike likely?" --as-of 2022-06-12 --fail news_search:primary
python -m volsentinel evaluate --start 2023-01-02 --out results/eval_2023_2025.json
python -m volsentinel tune                       # fit warning gate on 2021-22, one held-out check
python -m volsentinel mcp                        # MCP server (stdio): forecast, runs, evidence, tree, trace
python -m pytest -q
```

The web UI shows:

- the HAR regime probabilities against the naive baseline;
- the volatility scale with the regime thresholds;
- the Tree of Thought, with the selected, alternative, kept, protected-tail and pruned nodes;
- the cited explanation, where you click a citation to see its evidence;
- the precedents that passed or were rejected, with their scores;
- the human-review reasons;
- the full trace of every agent step.

### API

| Method | Path | Body or notes |
|---|---|---|
| POST | `/api/forecast` | `{question, as_of?, mode: replay|live, headlines?, simulate_failures?, trigger: interactive|scheduled}` |
| GET | `/api/runs`, `/api/runs/<id>` | saved answers plus shared state |
| GET | `/api/health`, `/api/dates`, `/api/metrics` | |

### Optional Claude layer

The system runs end to end without a language model. Tool choice, scoring and guardrails are
deterministic. If you install `anthropic` and set `ANTHROPIC_API_KEY`, Claude does two jobs:

- it drafts the analyst's explanation;
- it gives the critic's clarity and specificity judgement.

The default model is `claude-opus-5-5`; override it with `VOLSENTINEL_MODEL`. Server-side
refusal fallbacks are enabled. LLM output must still pass the deterministic citation and language
checks. If it fails, the analyst rewrites the explanation from the template in round two.

## Evaluation (replayed weeks 2023-01-02 to 2025-11-24)

`results/eval_2023_2025.json` and `.csv` hold the results. Each week runs exactly as it would have
on that day:

- HAR is refit on past data only;
- a precedent must be at least 30 days old;
- memory only sees forecasts whose seven-day window has already closed.

See that file for the current numbers. Headline results:

* **Correctness.** HAR has a Brier score of about 0.55, against 0.63 for the naive
  "last week persists" baseline. This reproduces the Checkpoint 3 backtest. HAR is worse than
  naive on the few stressed weeks.
* **Tree of Thought vs a single chain of thought.** ToT matches the realised direction more
  often. Both have high false-alarm rates on warnings: only 11 of 152 weeks were stressed.
* **Groundedness, reliability, efficiency.** All approved explanations are fully grounded.
  The p95 latency is under 1 s without an LLM, and runs use at most 11 of the 15 tool calls.

### Warning-gate tuning (validation 2021-2022, held-out test 2023-2025)

`python -m volsentinel tune` fits a **warning gate** on the 2021-2022 validation weeks only. The
gate keeps a "warning" or "severe" level only when the branch is strong enough, and otherwise
lowers it to "watch". The tree search itself does not change. The gate has four thresholds:

- the scenario's score;
- its evidence support;
- a same-family precedent *or* HAR P(Stressed) above a level;
- the number of risk headlines.

The objective is false alarms + 5 × stressed weeks missed. The tuner freezes the thresholds,
then looks once at the held-out weeks. It adopts the gate only if the cost there is lower and no
correct warning is lost. This is the same rule Checkpoint 3.1 applies to the precedent blend.

| | Validation before | Validation after | Held-out before | Held-out after |
|---|---|---|---|---|
| Warnings | 19 | 11 | 28 | 22 |
| False alarms | 15 | 7 | 26 | 21 |
| Stressed weeks warned | 4 of 14 | 4 of 14 | 2 of 11 | 1 of 11 |
| Cost | 65 | 57 | 71 | 71 |

The fitted gate keeps a warning only when P(Stressed) ≥ 0.25 or a same-family precedent passed,
and there are at least 6 risk headlines. It halved false alarms on validation but did not improve
the held-out weeks, so it is **not adopted**. `volsentinel/tuned.json` records the fit with
`"adopted": false`, and the defaults stay in force.

Gating can only remove warnings. It cannot raise recall: 7 of the 11 held-out stressed weeks had
no raw warning at all. The next gains need better early signals, such as exchange outflows or
options skew, rather than more threshold search. Do not re-tune against the held-out weeks.

## Limits

* Without an exchange-balance feed, "exchange run" scenarios rely on headlines, precedents,
  sentiment and drawdown.
* The project DB funding series and the live OKX funding series differ in scale. Z-scores are
  computed within each source.
* Some headlines in the CryptoPanic archive contain mis-encoded characters.
