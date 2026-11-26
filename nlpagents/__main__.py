"""Command line: python -m nlpagents <command>

  analyze  [FILE | -]  [--backend ollama|rules] [--single] [--json]
  tone     [FILE | -]                       quick rule-based tone check, no model needed
  benchmark [--backend ollama|rules] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import Orchestrator, single_prompt
from .llm import OllamaClient
from .tone import analyze_tone, tone_shifts


def _read(src: str) -> str:
    return sys.stdin.read() if src == "-" else Path(src).read_text(encoding="utf-8")


def _client(backend: str):
    if backend == "rules":
        return None
    client = OllamaClient()
    if not client.available():
        sys.exit(f"Ollama is not running or '{client.model}' is not pulled.\n"
                 f"  Start it and run:  ollama pull {client.model}\n  Or use --backend rules (offline).")
    return client


def print_report(rep) -> None:
    print(f"\n=== SUMMARY ({rep.pipeline}, {rep.backend}, {rep.iterations} round(s), "
          f"{'approved' if rep.approved else 'not approved'}) ===\n{rep.summary}\n")
    print("KEY POINTS:")
    for p in rep.key_points:
        print(f"  • {p}")
    t = rep.tone
    print(f"\nTONE: {t.get('sentiment')} · {t.get('formality')} · {', '.join(t.get('labels', []))}")
    if t.get("explanation"):
        print(f"  {t['explanation']}")
    if rep.recommendations:
        print("\nSUGGESTIONS:")
        for r in rep.recommendations:
            print(f"  • {r}")
    for i, rv in enumerate(rep.reviews, 1):
        print(f"\nReview round {i}: score {rv.get('score')}/10, approved={rv.get('approved')}")
        for issue in rv.get("issues", []):
            print(f"  - [{issue.get('type')}] {issue.get('detail')}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m nlpagents")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("src", nargs="?", default="-")
    a.add_argument("--backend", choices=["ollama", "rules"], default="ollama")
    a.add_argument("--single", action="store_true", help="single-prompt baseline instead of the agents")
    a.add_argument("--rounds", type=int, default=2)
    a.add_argument("--json", action="store_true")
    t = sub.add_parser("tone")
    t.add_argument("src", nargs="?", default="-")
    b = sub.add_parser("benchmark")
    b.add_argument("--backend", choices=["ollama", "rules"], default="ollama")
    b.add_argument("--limit", type=int)
    args = p.parse_args(argv)

    if args.cmd == "tone":
        rep = analyze_tone(_read(args.src))
        out = rep.to_dict()
        out["tone_shifts"] = tone_shifts(rep)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif args.cmd == "analyze":
        client = _client(args.backend)
        text = _read(args.src)
        rep = single_prompt(client, text) if args.single else Orchestrator(client, args.rounds).run(text)
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False)) if args.json else print_report(rep)
    elif args.cmd == "benchmark":
        from .benchmark import run_benchmark, to_markdown
        res = run_benchmark(_client(args.backend), args.limit)
        out = Path("results")
        out.mkdir(exist_ok=True)
        (out / f"benchmark_{res['backend'].replace(':', '_')}.json").write_text(json.dumps(res, indent=2))
        md = to_markdown(res)
        (out / f"benchmark_{res['backend'].replace(':', '_')}.md").write_text(md)
        print("\n" + md)


if __name__ == "__main__":
    main()
