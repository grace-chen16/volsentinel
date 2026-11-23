"""Command line.

  python -m volsentinel ask "Is a volatility spike likely this week?" --as-of 2022-11-07
  python -m volsentinel ask --live
  python -m volsentinel ask --live --scheduled   # daily brief: flagged answers are held for review
  python -m volsentinel ingest                   # fetch live feeds into live_data/live_store.db
  python -m volsentinel serve --port 8000
  python -m volsentinel evaluate --start 2023-01-02 --out results/eval.json
  python -m volsentinel score            # score forecasts whose 7-day window has closed
  python -m volsentinel mcp              # MCP server over stdio (read-only state + forecast tool)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path


def _print_answer(a: dict, trace: bool):
    if a["status"] == "failed":
        print(a["headline"]); print(a["explanation"]); return
    f = a["forecast"]
    print(f"\nVolSentinel  as of {a['asof']} ({a['mode']})  run {a['run_id']}")
    print("=" * 78)
    print(a["headline"])
    for r, p in f["probabilities"].items():
        print(f"  {r:<9} {'#' * round(p * 40):<40} {p:5.0%}")
    print(f"\nstatus {a['status']} | confidence {a['confidence']} | review: "
          f"{a['human_review']['release'] if a['human_review']['required'] else 'not required'}")
    for r in a["human_review"]["reasons"]:
        print(f"  - {r}")
    print("\n" + textwrap.fill(a["explanation"], 100))
    if a["precedents"] or a["rejected_precedents"]:
        print("\nprecedents:")
        for p in a["precedents"]:
            print(f"  + {p['title']} ({p['date']}) sem {p['semantic_sim']:.2f} cond {p['condition_sim']:.2f} "
                  f"-> next 7d {p['outcome']['next_7d_regime']}")
        for p in a["rejected_precedents"]:
            print(f"  - {p['title']} ({p['date']}): {p['reject_reason']}")
    m = a["metrics"]
    print(f"\n{m['tool_calls']} tool calls, {m['node_evals']} node evaluations, {m['latency_s']}s")
    if trace:
        print("\ntrace:")
        for e in a["trace"]:
            print(f"  {e['step']:>3} {e['agent']:<16} {e['kind']:<8} {e['text'][:140]}")
    print("\n" + a["disclaimer"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="volsentinel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question", nargs="?", default="Is a volatility spike likely this week?")
    a.add_argument("--as-of")
    a.add_argument("--live", action="store_true")
    a.add_argument("--headline", action="append", default=[])
    a.add_argument("--fail", action="append", default=[], help="simulate a tool failure, e.g. news_search:primary")
    a.add_argument("--trace", action="store_true")
    a.add_argument("--json", action="store_true")
    a.add_argument("--no-news", action="store_true", help="skip loading the news archive")
    a.add_argument("--scheduled", action="store_true",
                   help="run as the scheduled daily brief: answers that trigger review are held, not released")
    sub.add_parser("ingest", help="fetch live feeds and store them, without forecasting")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("evaluate")
    e.add_argument("--start", default="2023-01-02")
    e.add_argument("--end")
    e.add_argument("--limit", type=int)
    e.add_argument("--out", default="results/eval_2023_2025.json")
    t = sub.add_parser("tune")
    t.add_argument("--start", default="2021-01-04")
    t.add_argument("--end", default="2022-12-26")
    sub.add_parser("score")
    sub.add_parser("mcp")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.cmd in ("serve", "evaluate", "tune") else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    if args.cmd == "serve":
        from .api import app, ctx
        import socket
        with socket.socket() as probe:     # fail clearly, before loading data, if the port is taken
            try:
                probe.bind((args.host, args.port))
            except OSError:
                print(f"Port {args.port} is already in use or blocked on this computer. "
                      f"Try another, for example: python -m volsentinel serve --port 8050", file=sys.stderr)
                return 1
        ctx()                      # load data before accepting requests
        print(f"VolSentinel UI: http://{args.host}:{args.port}/", flush=True)
        app.run(host=args.host, port=args.port, threaded=True)
        return
    if args.cmd == "mcp":
        from .mcp_server import mcp
        mcp.run()
        return

    from .context import Context
    no_news = getattr(args, "no_news", False) or args.cmd in ("ingest", "score")
    ctx = Context(load_news=False if no_news else None)
    if args.cmd == "ask":
        from .agents.orchestrator import run_forecast
        ans = run_forecast(ctx, args.question, args.as_of, "live" if args.live else "replay", args.headline, args.fail,
                           trigger="scheduled" if args.scheduled else "interactive")
        if args.json:
            print(json.dumps(ans, indent=1, default=str))
        else:
            _print_answer(ans, args.trace)
        held = ans.get("human_review", {}).get("release") == "hold for human review"
        if args.scheduled and held:
            print("\nHELD FOR REVIEW: this scheduled brief must be checked by a person before release.",
                  file=sys.stderr if args.json else sys.stdout)
        return 2 if held else 0
    elif args.cmd == "ingest":
        print(json.dumps(ctx.ingest(), indent=1, default=str))
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        rep = evaluate(ctx, args.start, args.end, out=Path(args.out), limit=args.limit)
        print(json.dumps(rep, indent=1, default=str))
    elif args.cmd == "tune":
        from .tune import tune
        print(json.dumps(tune(ctx, args.start, args.end), indent=1, default=str))
    elif args.cmd == "score":
        n = ctx.memory.score_due(ctx.stored_market())    # archive + stored live days
        print(f"scored {n} forecast(s)")
        print(json.dumps(ctx.memory.metrics(), indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main() or 0)
