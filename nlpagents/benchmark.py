"""Benchmark: multi-agent pipeline vs single-prompt baseline.

Dataset: ``data/benchmark.jsonl`` – short texts of many kinds (reviews, complaints,
formal letters, news, research, social posts, sarcasm, rude forum posts) labelled
with sentiment, formality and the key points a good summary should contain.

Metrics
* sentiment accuracy, formality accuracy
* key-point recall – share of reference key points found in the summary/key points
  (content-word overlap ≥ 50 %)
* lexical support – share of output sentences whose content words mostly (≥ 60 %)
  appear in the source; a cheap proxy for faithfulness / hallucination
* seconds per text
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .agents import Orchestrator, Report, sentence_split, single_prompt
from .llm import LLMClient

DATA = Path(__file__).resolve().parent.parent / "data" / "benchmark.jsonl"
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "be", "by",
        "it", "that", "this", "after", "before", "from", "at", "as", "but", "not", "no"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


def keypoint_recall(report: Report, refs: list[str]) -> float:
    pool = [report.summary] + list(report.key_points)
    out_words = set().union(*(_content(p) for p in pool))
    hits = 0
    for ref in refs:
        rw = _content(ref)
        if rw and len(rw & out_words) / len(rw) >= 0.5:
            hits += 1
    return hits / len(refs)


def lexical_support(report: Report, source: str) -> float:
    src = _content(source)
    sents = sentence_split(report.summary) + list(report.key_points)
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        w = _content(s)
        ok += (not w) or len(w & src) / len(w) >= 0.6
    return ok / len(sents)


@dataclass
class Row:
    id: str
    pipeline: str
    sentiment_ok: bool
    formality_ok: bool
    recall: float
    support: float
    seconds: float
    predicted: dict


def load_dataset(path: Path = DATA) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_benchmark(client: LLMClient | None, limit: int | None = None, max_rounds: int = 2) -> dict:
    data = load_dataset()[:limit]
    orch = Orchestrator(client, max_rounds=max_rounds)
    rows: list[Row] = []
    for item in data:
        for name, fn in (("multi-agent", orch.run), ("single-prompt", lambda t: single_prompt(client, t))):
            try:
                rep = fn(item["text"])
            except Exception as exc:  # a malformed reply counts as a miss, not a crash
                print(f"  {item['id']} [{name}] failed: {exc}")
                rep = Report("", [], {}, [], [], {}, {}, pipeline=name)
            rows.append(Row(item["id"], name, rep.tone.get("sentiment") == item["sentiment"],
                            rep.tone.get("formality") == item["formality"], keypoint_recall(rep, item["key_points"]),
                            lexical_support(rep, item["text"]), rep.seconds,
                            {"tone": rep.tone, "summary": rep.summary, "iterations": rep.iterations}))
        print(f"  done {item['id']}")
    summary = {}
    for name in ("multi-agent", "single-prompt"):
        rs = [r for r in rows if r.pipeline == name]
        n = len(rs)
        summary[name] = {"sentiment_accuracy": sum(r.sentiment_ok for r in rs) / n,
                         "formality_accuracy": sum(r.formality_ok for r in rs) / n,
                         "keypoint_recall": sum(r.recall for r in rs) / n,
                         "lexical_support": sum(r.support for r in rs) / n,
                         "seconds_per_text": sum(r.seconds for r in rs) / n, "n": n}
    return {"backend": client.name if client else "rules", "summary": summary, "rows": [r.__dict__ for r in rows]}


def to_markdown(result: dict) -> str:
    lines = [f"Backend: `{result['backend']}`", "",
             "| Pipeline | Sentiment acc. | Formality acc. | Key-point recall | Lexical support | s / text |",
             "|---|---|---|---|---|---|"]
    for name, s in result["summary"].items():
        lines.append(f"| {name} | {s['sentiment_accuracy']:.0%} | {s['formality_accuracy']:.0%} | "
                     f"{s['keypoint_recall']:.0%} | {s['lexical_support']:.0%} | {s['seconds_per_text']:.1f} |")
    return "\n".join(lines)
