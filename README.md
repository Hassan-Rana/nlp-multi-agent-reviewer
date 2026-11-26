# 🧠 Multi-Agent NLP Synthesizer & Reviewer (with tone analysis)

A multi-agent text-analysis pipeline running on **Llama 3.1 (8B)** through [Ollama](https://ollama.com).
Specialised agents analyse a text, judge its **tone**, write a report and **check each other's work**
in a review loop. It is benchmarked against a single-prompt baseline.

```
 text ─┬─► Analyst agent ─── summary, key points, entities, claims
       └─► Tone agent ─────── sentiment, formality, politeness, confidence, emotions,
              ▲                tone shifts, evidence quotes, suggestions
              │ rule-based tone signals (evidence)
              ▼
       Synthesizer agent ──► draft report ──► Reviewer agent ── approved? ──► final report
              ▲                                   │ no: issues (unsupported claim,
              └──────────── revise ◄──────────────┘     missing point, wrong tone)
```

**Cross-agent validation:** besides the Reviewer's judgement, an independent rule-based tone checker
(lexicons with negation and intensifier handling, formality and politeness markers, hedges, all-caps
and insult detection) validates the Tone agent. If the model calls an angry complaint "positive", the
draft goes back for revision with the evidence attached.

## What the tone check reports

| Dimension | Values |
|---|---|
| Sentiment | positive · negative · neutral · mixed (with a −1…+1 score) |
| Formality | formal · informal (0–1 score: contractions, slang, emoji, word length, formal connectives) |
| Politeness | polite · neutral · impolite |
| Confidence | assertive · balanced · hedging (*maybe, might, I think* vs *clearly, definitely*) |
| Aggression | 0–1 (all-caps, repeated "!!", insults) |
| Emotions | joy, anger, sadness, fear, surprise, trust |
| Tone shifts | sentences where the sentiment flips |
| Suggestions | how to adjust the tone for the purpose of the text |

## Run it

```bash
# 1. the model
ollama pull llama3.1:8b          # https://ollama.com/download

# 2. the project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m nlpagents analyze examples/complaint.txt             # multi-agent pipeline
python -m nlpagents analyze examples/report.txt --single       # single-prompt baseline
python -m nlpagents tone examples/complaint.txt                # instant rule-based tone check
python -m nlpagents analyze examples/report.txt --backend rules  # fully offline, no model
python -m nlpagents benchmark                                  # multi-agent vs single prompt
streamlit run app.py                                           # web UI
python -m pytest -q                                            # tests (no model needed)
```

`OLLAMA_MODEL` and `OLLAMA_HOST` environment variables change the model and server.

Example (offline `rules` backend):

```
TONE: negative · informal · negative, informal, impolite, aggressive, anger

SUGGESTIONS:
  • Remove all-caps, repeated exclamation marks and insults; state the problem and the request calmly.
  • Add a courteous opening and a clear, polite request.
```

## Benchmark

`data/benchmark.jsonl` holds 24 short texts of different kinds (reviews, complaints, formal letters,
news, research summaries, social posts, sarcasm, a rude forum post, incident reports), each labelled
with sentiment, formality and the key points a good summary should contain.

| Metric | Meaning |
|---|---|
| Sentiment / formality accuracy | tone labels match the reference |
| Key-point recall | share of reference key points covered by the summary and key points |
| Lexical support | share of output sentences whose content words come from the source (a proxy for hallucination) |
| s / text | latency |

Offline `rules` backend (no LLM), for reference: [`results/benchmark_rules.md`](results/benchmark_rules.md).
Run `python -m nlpagents benchmark` with Ollama to produce `results/benchmark_ollama_llama3.1_8b.md`
and compare the multi-agent pipeline with the single-prompt baseline.

## Project layout

```
nlpagents/
  llm.py        Ollama client, scripted client for tests, robust JSON parsing
  prompts.py    one system prompt per agent + the single-prompt baseline
  tone.py       rule-based tone analyser (sentiment, formality, politeness, hedging, emotions, shifts)
  agents.py     Analyst, Tone, Synthesizer, Reviewer agents, cross-check, orchestrator, baseline
  benchmark.py  dataset, metrics, comparison table
app.py          Streamlit UI with the full agent trace
data/benchmark.jsonl  labelled evaluation texts
tests/          pipeline tests with a scripted LLM (no model needed)
```
