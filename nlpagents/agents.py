"""The agents and the orchestrator.

Pipeline (multi-agent):

    text ─┬─► Analyst  (summary, key points, entities, claims)
          └─► Tone     (sentiment, formality, politeness, confidence, emotions, evidence)
                 │      ▲ rule-based tone signals as evidence
                 ▼
            Synthesizer ──► draft report ──► Reviewer ──approved?──► final report
                 ▲                              │ no (issues)
                 └──────── revise ◄─────────────┘   (up to N rounds)

Cross-agent validation: besides the Reviewer's own judgement, the orchestrator compares
the Tone agent's labels with the independent rule-based tone checker and adds a "tone"
issue when they clearly disagree (e.g. positive vs negative).

Every agent has an LLM implementation and an offline ``rules`` implementation, so
the pipeline runs with or without a model.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

from . import prompts
from .llm import LLMClient, parse_json
from .tone import ToneReport, analyze_tone, sentence_split, tone_shifts

STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being it its this
that these those i you he she we they me him her us them my your our their his not no so than then there here
have has had do does did will would can could should may might must just also very really about into over
after before more most such only own same too s t don't what which who whom when where why how all any each few
other some""".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2]


def support(sentence: str, source: str) -> float:
    """Share of a sentence's content words that appear in the source (0–1)."""
    ws = _words(sentence)
    if not ws:
        return 1.0
    src = set(_words(source))
    return sum(w in src for w in ws) / len(ws)


# ---------------------------------------------------------------- Analyst
def analyst_rules(text: str) -> dict:
    sents = sentence_split(text)
    freq = Counter(_words(text))
    scored = [(sum(freq[w] for w in _words(s)) / (len(_words(s)) + 1) ** 0.5, i, s) for i, s in enumerate(sents)]
    k = max(1, min(3, len(sents) // 3 or 1))
    top = sorted(sorted(scored, reverse=True)[:k], key=lambda x: x[1])
    ents = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,})\b", text)
    claims = [s for s in sents if re.search(r"\d|\b(is|are|was|will|has|have)\b", s)][:5]
    return {"summary": " ".join(s for _, _, s in top),
            "key_points": [s for _, _, s in sorted(scored, reverse=True)[:min(4, len(sents))]],
            "entities": sorted(set(ents))[:10], "claims": claims}


def analyst_llm(client: LLMClient, text: str) -> dict:
    return parse_json(client.chat(prompts.ANALYST, f"TEXT:\n{text}"))


# ---------------------------------------------------------------- Tone
def tone_rules(rules: ToneReport) -> dict:
    return {"sentiment": rules.sentiment, "formality": rules.formality, "politeness": rules.politeness,
            "confidence": rules.confidence, "emotions": list(rules.emotions), "overall_tone": rules.overall,
            "evidence": rules.evidence.get("positive_words", [])[:3] + rules.evidence.get("negative_words", [])[:3],
            "tone_shifts": tone_shifts(rules), "suggestions": tone_suggestions(rules)}


def tone_llm(client: LLMClient, text: str, rules: ToneReport) -> dict:
    signals = {k: v for k, v in rules.to_dict().items() if k != "sentences"}
    return parse_json(client.chat(prompts.TONE, f"TEXT:\n{text}\n\nRULE-BASED SIGNALS:\n{json.dumps(signals)}"))


def tone_suggestions(r: ToneReport) -> list[str]:
    out = []
    if r.aggression >= 0.5:
        out.append("Remove all-caps, repeated exclamation marks and insults; state the problem and the request calmly.")
    if r.politeness == "impolite":
        out.append("Add a courteous opening and a clear, polite request.")
    if r.confidence == "hedging":
        out.append("Cut some hedges (maybe, might, I think) where you are actually sure.")
    if r.formality == "informal":
        out.append("For professional readers, replace slang and contractions with full forms.")
    return out


# ---------------------------------------------------------------- Synthesizer
def synthesize_rules(analysis: dict, tone: dict, feedback: list[dict] | None = None, source: str = "") -> dict:
    summary_sents = sentence_split(analysis["summary"])
    points = list(analysis["key_points"])
    for issue in feedback or []:
        if issue["type"] == "unsupported":
            summary_sents = [s for s in summary_sents if s not in issue["detail"]]
            points = [p for p in points if p not in issue["detail"]]
        if issue["type"] == "missing" and "sentence" in issue:
            summary_sents.append(issue["sentence"])
    labels = tone.get("overall_tone", [])
    return {"summary": " ".join(summary_sents), "key_points": points,
            "tone": {"sentiment": tone["sentiment"], "formality": tone["formality"], "labels": labels,
                     "explanation": f"The text reads as {', '.join(labels) or tone['sentiment']}."},
            "recommendations": tone.get("suggestions", [])}


def synthesize_llm(client: LLMClient, analysis: dict, tone: dict, feedback: list[dict] | None,
                   draft: dict | None, source: str) -> dict:
    msg = f"SOURCE TEXT:\n{source}\n\nANALYST:\n{json.dumps(analysis)}\n\nTONE:\n{json.dumps(tone)}"
    if feedback:
        msg += f"\n\nPREVIOUS DRAFT:\n{json.dumps(draft)}\n\nREVIEWER ISSUES TO FIX:\n{json.dumps(feedback)}"
    return parse_json(client.chat(prompts.SYNTHESIZER, msg))


# ---------------------------------------------------------------- Reviewer
def review_rules(source: str, draft: dict, rules: ToneReport) -> dict:
    issues = []
    for s in sentence_split(draft.get("summary", "")) + list(draft.get("key_points", [])):
        if support(s, source) < 0.6:
            issues.append({"type": "unsupported", "detail": s})
    freq = Counter(_words(source))
    covered = set(_words(draft.get("summary", "") + " " + " ".join(draft.get("key_points", []))))
    missing = [w for w, c in freq.most_common(5) if c >= 2 and w not in covered]
    if missing:
        best = max(sentence_split(source), key=lambda s: sum(w in s.lower() for w in missing))
        issues.append({"type": "missing", "detail": f"Key terms not covered: {', '.join(missing)}",
                       "sentence": best})
    score = max(1, 10 - 2 * len(issues))
    return {"approved": not issues, "score": score, "issues": issues}


def review_llm(client: LLMClient, source: str, draft: dict, rules: ToneReport) -> dict:
    signals = {"sentiment": rules.sentiment, "formality": rules.formality, "politeness": rules.politeness,
               "evidence": rules.evidence}
    out = parse_json(client.chat(prompts.REVIEWER, f"SOURCE:\n{source}\n\nDRAFT REPORT:\n{json.dumps(draft)}"
                                                   f"\n\nRULE-BASED TONE SIGNALS:\n{json.dumps(signals)}"))
    out.setdefault("issues", [])
    out["approved"] = bool(out.get("approved")) and not any(i.get("type") == "unsupported" for i in out["issues"])
    return out


def cross_check_tone(tone: dict, rules: ToneReport) -> list[dict]:
    """Independent validation of the Tone agent against the rule-based checker."""
    issues = []
    a, b = tone.get("sentiment"), rules.sentiment
    if {a, b} == {"positive", "negative"} and abs(rules.sentiment_score) > 0.5:
        issues.append({"type": "tone", "detail": f"Tone agent says '{a}' but the text's wording is clearly '{b}' "
                       f"(evidence: {rules.evidence['positive_words'] + rules.evidence['negative_words']}). "
                       "Re-check, or explain the sarcasm if that is the reason."})
    if tone.get("politeness") == "polite" and rules.politeness == "impolite" and rules.aggression >= 0.5:
        issues.append({"type": "tone", "detail": "Labelled polite, but the text has insults or shouting."})
    return issues


# ---------------------------------------------------------------- Orchestrator
@dataclass
class Report:
    summary: str
    key_points: list[str]
    tone: dict
    recommendations: list[str]
    entities: list[str]
    tone_details: dict
    rule_tone: dict
    reviews: list[dict] = field(default_factory=list)
    iterations: int = 1
    approved: bool = False
    backend: str = "rules"
    pipeline: str = "multi-agent"
    seconds: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Orchestrator:
    def __init__(self, client: LLMClient | None = None, max_rounds: int = 2):
        self.client = client
        self.max_rounds = max_rounds

    @property
    def backend(self) -> str:
        return self.client.name if self.client else "rules"

    def _step(self, trace: list, agent: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        trace.append({"agent": agent, "seconds": round(time.perf_counter() - t0, 3), "output": out})
        return out

    def run(self, text: str) -> Report:
        t0 = time.perf_counter()
        trace: list[dict] = []
        rules = analyze_tone(text)
        c = self.client
        if c:
            analysis = self._step(trace, "analyst", analyst_llm, c, text)
            tone = self._step(trace, "tone", tone_llm, c, text, rules)
        else:
            analysis = self._step(trace, "analyst", analyst_rules, text)
            tone = self._step(trace, "tone", tone_rules, rules)
        draft, reviews, feedback = None, [], None
        for round_no in range(1, self.max_rounds + 2):
            if c:
                draft = self._step(trace, "synthesizer", synthesize_llm, c, analysis, tone, feedback, draft, text)
                review = self._step(trace, "reviewer", review_llm, c, text, draft, rules)
            else:
                draft = self._step(trace, "synthesizer", synthesize_rules, analysis, tone, feedback, text)
                review = self._step(trace, "reviewer", review_rules, text, draft, rules)
            extra = cross_check_tone(draft.get("tone", {}), rules)
            if extra:
                review["issues"] = review.get("issues", []) + extra
                review["approved"] = False
            reviews.append(review)
            if review["approved"] or round_no > self.max_rounds:
                break
            feedback = review["issues"]
        tone_out = draft.get("tone", {})
        return Report(summary=draft.get("summary", ""), key_points=draft.get("key_points", []), tone=tone_out,
                      recommendations=draft.get("recommendations", []), entities=analysis.get("entities", []),
                      tone_details=tone, rule_tone=rules.to_dict(), reviews=reviews, iterations=len(reviews),
                      approved=reviews[-1]["approved"], backend=self.backend,
                      seconds=round(time.perf_counter() - t0, 2), trace=trace)


def single_prompt(client: LLMClient | None, text: str) -> Report:
    """Baseline: one prompt does everything, no review loop."""
    t0 = time.perf_counter()
    rules = analyze_tone(text)
    if client:
        out = parse_json(client.chat(prompts.SINGLE_PROMPT, f"TEXT:\n{text}"))
    else:
        a = analyst_rules(text)
        out = {"summary": a["summary"], "key_points": a["key_points"],
               "tone": {"sentiment": rules.sentiment, "formality": rules.formality, "labels": rules.overall,
                        "explanation": ""}, "recommendations": []}
    return Report(summary=out.get("summary", ""), key_points=out.get("key_points", []), tone=out.get("tone", {}),
                  recommendations=out.get("recommendations", []), entities=[], tone_details={},
                  rule_tone=rules.to_dict(), backend=client.name if client else "rules", pipeline="single-prompt",
                  approved=True, seconds=round(time.perf_counter() - t0, 2))
