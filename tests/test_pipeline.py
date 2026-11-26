import json

import pytest

from nlpagents.agents import Orchestrator, cross_check_tone, single_prompt, support
from nlpagents.benchmark import keypoint_recall, load_dataset, run_benchmark
from nlpagents.llm import ScriptedClient, parse_json
from nlpagents.tone import analyze_tone, tone_shifts

COMPLAINT = ("This is UNACCEPTABLE!!! I ordered the blender three weeks ago and it still hasn't arrived. "
             "Your support team is useless and nobody answers the phone. I want a full refund today.")


def test_tone_complaint_is_negative_informal_aggressive():
    r = analyze_tone(COMPLAINT)
    assert r.sentiment == "negative"
    assert r.formality == "informal"
    assert r.aggression >= 0.5 and "aggressive" in r.overall
    assert "UNACCEPTABLE" in r.evidence["all_caps"]


def test_tone_negation_and_formal_letter():
    assert analyze_tone("The service was not good and the room was not clean.").sentiment == "negative"
    formal = analyze_tone("Dear Sir, Please find attached the report. Furthermore, kindly confirm receipt. "
                          "Sincerely, A. Smith")
    assert formal.formality == "formal" and formal.politeness == "polite"


def test_tone_hedging_and_shift():
    r = analyze_tone("The results might suggest a small effect. Perhaps it is noise. I think we probably need more data.")
    assert r.confidence == "hedging"
    shifts = tone_shifts(analyze_tone("The food was delicious. The service was terrible."))
    assert shifts


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": {"b": 2}} hope this helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_rules_pipeline_runs_offline():
    rep = Orchestrator(None).run(COMPLAINT)
    assert rep.backend == "rules"
    assert rep.summary and rep.key_points
    assert rep.tone["sentiment"] == "negative"
    assert rep.reviews and rep.iterations >= 1
    assert all(support(s, COMPLAINT) >= 0.6 for s in rep.key_points)


def test_cross_check_flags_wrong_tone():
    issues = cross_check_tone({"sentiment": "positive"}, analyze_tone(COMPLAINT))
    assert issues and issues[0]["type"] == "tone"


def _llm_script():
    analyst = json.dumps({"summary": "A customer's blender has not arrived after three weeks.",
                          "key_points": ["Blender ordered three weeks ago has not arrived", "Wants a full refund"],
                          "entities": [], "claims": ["support is useless"]})
    tone = json.dumps({"sentiment": "positive", "formality": "informal", "politeness": "impolite",
                       "confidence": "assertive", "emotions": ["anger"], "overall_tone": ["angry"],
                       "evidence": ["UNACCEPTABLE"], "tone_shifts": [], "suggestions": ["Stay calm"]})
    draft_bad = json.dumps({"summary": "A customer's blender has not arrived after three weeks.",
                            "key_points": ["Blender not arrived", "Wants a refund"],
                            "tone": {"sentiment": "positive", "formality": "informal", "labels": ["angry"],
                                     "explanation": "x"}, "recommendations": []})
    draft_good = json.dumps({"summary": "A customer's blender has not arrived after three weeks and they want a refund.",
                             "key_points": ["Blender not arrived", "Wants a refund"],
                             "tone": {"sentiment": "negative", "formality": "informal", "labels": ["angry", "demanding"],
                                      "explanation": "Shouting and blunt demands."}, "recommendations": ["Stay calm"]})
    review_ok = json.dumps({"approved": True, "score": 9, "issues": []})
    return ScriptedClient({"You are the ANALYST": [analyst], "You are the TONE": [tone],
                           "You are the SYNTHESIZER": [draft_bad, draft_good], "You are the REVIEWER": [review_ok]})


def test_llm_pipeline_revises_after_cross_check():
    client = _llm_script()
    rep = Orchestrator(client, max_rounds=2).run(COMPLAINT)
    assert rep.iterations == 2                     # first draft rejected by tone cross-check
    assert rep.reviews[0]["approved"] is False
    assert any(i["type"] == "tone" for i in rep.reviews[0]["issues"])
    assert rep.approved and rep.tone["sentiment"] == "negative"
    synth_calls = [u for k, u in client.log if k == "You are the SYNTHESIZER"]
    assert "REVIEWER ISSUES TO FIX" in synth_calls[1]   # feedback was passed to the revision


def test_single_prompt_baseline_with_llm():
    out = json.dumps({"summary": "s", "key_points": ["k"], "tone": {"sentiment": "negative", "formality": "informal",
                                                                    "labels": [], "explanation": ""},
                      "recommendations": []})
    rep = single_prompt(ScriptedClient({"expert text analyst": [out]}), COMPLAINT)
    assert rep.pipeline == "single-prompt" and rep.tone["sentiment"] == "negative"


def test_benchmark_rules_backend():
    data = load_dataset()
    assert len(data) >= 20 and {"text", "sentiment", "formality", "key_points"} <= set(data[0])
    res = run_benchmark(None, limit=5)
    s = res["summary"]["multi-agent"]
    assert 0 <= s["sentiment_accuracy"] <= 1 and s["n"] == 5


def test_keypoint_recall():
    from nlpagents.agents import Report
    rep = Report("The tram line will be extended by four kilometres.", [], {}, [], [], {}, {})
    assert keypoint_recall(rep, ["tram line extended by four kilometres", "funded by EU"]) == 0.5
