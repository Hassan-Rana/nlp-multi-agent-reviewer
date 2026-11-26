"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from nlpagents.agents import Orchestrator, single_prompt
from nlpagents.llm import OllamaClient
from nlpagents.tone import analyze_tone, tone_shifts

st.set_page_config(page_title="Multi-Agent Text Reviewer", page_icon="🧠", layout="wide")
st.title("🧠 Multi-agent synthesizer & reviewer")
st.caption("Analyst · Tone · Synthesizer · Reviewer agents on Llama 3.1 8B (Ollama), with a rule-based tone "
           "checker for cross-validation.")

client = OllamaClient()
ollama_ok = client.available()
backend = st.sidebar.radio("Backend", ["ollama", "rules"], index=0 if ollama_ok else 1,
                           help="'rules' runs offline without a language model.")
if backend == "ollama" and not ollama_ok:
    st.sidebar.error(f"Ollama not reachable or model missing. Run `ollama pull {client.model}`.")
pipeline = st.sidebar.radio("Pipeline", ["multi-agent", "single-prompt baseline"])
rounds = st.sidebar.slider("Max review rounds", 0, 3, 2)

text = st.text_area("Text to analyse", height=200,
                    placeholder="Paste an email, review, report, social post…")

col1, col2 = st.columns(2)
quick = col1.button("Quick tone check (instant)")
run = col2.button("Run agents", type="primary", disabled=not text.strip())

if quick and text.strip():
    r = analyze_tone(text)
    st.subheader("Tone")
    c = st.columns(5)
    c[0].metric("Sentiment", r.sentiment, f"{r.sentiment_score:+.2f}")
    c[1].metric("Formality", r.formality, f"{r.formality_score:.2f}")
    c[2].metric("Politeness", r.politeness)
    c[3].metric("Confidence", r.confidence)
    c[4].metric("Aggression", f"{r.aggression:.2f}")
    st.write("Emotions:", r.emotions or "none detected")
    for s in tone_shifts(r):
        st.info(s)
    st.json(r.evidence)

if run:
    llm = client if backend == "ollama" else None
    with st.spinner("Agents at work…"):
        rep = single_prompt(llm, text) if pipeline.startswith("single") else Orchestrator(llm, rounds).run(text)
    st.subheader("Summary")
    st.write(rep.summary)
    st.subheader("Key points")
    for p in rep.key_points:
        st.markdown(f"- {p}")
    st.subheader("Tone")
    t = rep.tone
    st.write(f"**{t.get('sentiment')}** · **{t.get('formality')}** · {', '.join(t.get('labels', []))}")
    if t.get("explanation"):
        st.caption(t["explanation"])
    if rep.recommendations:
        st.subheader("Suggestions")
        for r in rep.recommendations:
            st.markdown(f"- {r}")
    st.caption(f"{rep.pipeline} · {rep.backend} · {rep.iterations} round(s) · "
               f"{'approved' if rep.approved else 'not approved'} · {rep.seconds:.1f}s")
    with st.expander("Reviewer feedback"):
        st.json(rep.reviews)
    with st.expander("Full agent trace"):
        st.json(rep.trace)
