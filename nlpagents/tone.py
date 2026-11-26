"""Rule-based tone analysis.

Works offline and is fast. It is used in two ways:
1. as the tone agent in the offline ``rules`` backend, and
2. as independent evidence that the reviewer uses to cross-check the LLM tone agent
   (if the LLM says "positive" but the text is full of negative words, that is flagged).

Dimensions
* sentiment   – positive / negative / neutral / mixed (lexicon with negation and intensifiers)
* formality   – formal / informal (contractions, slang, emoji, exclamation, word length…)
* politeness  – polite / neutral / impolite
* confidence  – assertive / balanced / hedging
* emotions    – joy, anger, sadness, fear, surprise, trust (lexicon hits)
* aggression  – ALL CAPS, repeated "!!", insults, blunt commands
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

POSITIVE = set("""good great excellent amazing awesome fantastic wonderful love loved lovely like liked enjoy enjoyed
happy glad pleased delighted impressive impressed best better perfect brilliant outstanding superb nice helpful
useful success successful win won benefit beneficial improve improved improvement efficient reliable recommend
grateful thankful appreciate appreciated thrilled excited exciting positive strong smooth easy friendly
satisfied satisfying comfortable clean fast affordable valuable welcome congratulations proud remarkable
delicious tasty beautiful fresh spotless lovely promising effective favourable favorable progress growth gain gains solid robust""".split())
NEGATIVE = set("""bad poor terrible awful horrible hate hated dislike worst worse disappointing disappointed
disappoint annoying annoyed angry upset sad unhappy fail failed failure broken slow useless waste wasted problem
problems issue issues bug bugs error errors crash crashed delay delayed late expensive overpriced rude unhelpful
dirty difficult hard confusing confused frustrating frustrated unacceptable ridiculous pathetic disgusting
concern concerned worried worry risk risks loss losses decline declined drop dropped weak negative refund
complaint complain complained damage damaged painful unfortunately sorry regret mess scam inadequate
insufficient unreliable dangerous harmful toxic misleading""".split())
NEGATIONS = {"not", "no", "never", "n't", "hardly", "barely", "without", "nothing", "neither", "nor", "cannot"}
INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "totally": 1.5, "absolutely": 1.7,
                "incredibly": 1.7, "highly": 1.5, "super": 1.5, "quite": 1.2, "too": 1.3, "deeply": 1.5}
HEDGES = ["maybe", "perhaps", "possibly", "might", "may", "could", "seems", "seem", "appears", "i think",
          "i guess", "i believe", "probably", "somewhat", "sort of", "kind of", "not sure", "likely", "unclear",
          "it is possible", "suggests"]
BOOSTERS = ["definitely", "certainly", "clearly", "obviously", "undoubtedly", "always", "must", "will",
            "guarantee", "without doubt", "proven", "of course", "absolutely", "in fact"]
POLITE = ["please", "thank you", "thanks", "kindly", "would you", "could you", "appreciate", "grateful",
          "sorry", "apologies", "apologize", "apologise", "if possible", "would you mind", "dear", "regards",
          "sincerely", "with respect"]
IMPOLITE = ["stupid", "idiot", "idiots", "nonsense", "shut up", "useless", "pathetic", "ridiculous", "dumb",
            "incompetent", "clueless", "garbage", "trash", "worst", "what the hell", "wtf", "damn"]
SLANG = ["gonna", "wanna", "gotta", "kinda", "sorta", "lol", "lmao", "omg", "btw", "tbh", "idk", "imo",
         "yeah", "yep", "nope", "hey", "guys", "cool", "awesome", "dude", "stuff", "super", "pretty much",
         "ain't", "y'all", "u ", "ur ", "thx", "pls", "plz"]
FORMAL = ["therefore", "furthermore", "moreover", "however", "consequently", "regarding", "accordingly",
          "hereby", "pursuant", "nevertheless", "thus", "hence", "whereas", "in accordance", "we wish to",
          "please find", "kindly", "sincerely", "respectfully", "dear", "to whom it may concern"]
EMOTIONS = {
    "joy": "happy glad delighted joy love loved enjoy enjoyed excited thrilled wonderful great celebrate proud".split(),
    "anger": "angry furious outraged annoyed hate unacceptable ridiculous disgusting rage livid infuriating".split(),
    "sadness": "sad unhappy disappointed regret sorry miss lost loss heartbroken unfortunately grief".split(),
    "fear": "afraid scared worried worry fear risk risks anxious concern concerned threat dangerous panic".split(),
    "surprise": "surprised surprising unexpected suddenly wow shocked astonished amazed unbelievable".split(),
    "trust": "reliable trust trusted confident honest secure dependable recommend proven safe".split(),
}

INSULTS = {"stupid", "idiot", "idiots", "nonsense", "pathetic", "dumb", "incompetent", "clueless", "garbage", "trash"}

WORD_RE = re.compile(r"[A-Za-z']+|[!?]")


@dataclass
class ToneReport:
    sentiment: str
    sentiment_score: float  # -1 … +1
    formality: str
    formality_score: float  # 0 informal … 1 formal
    politeness: str
    confidence: str
    aggression: float  # 0 … 1
    emotions: dict[str, int]
    overall: list[str]
    evidence: dict[str, list[str]] = field(default_factory=dict)
    sentences: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    toks = []
    for t in WORD_RE.findall(text):
        t = t.lower()
        if t.endswith("n't"):
            toks += [t[:-3], "n't"]
        else:
            toks.append(t)
    return toks


def _count_phrases(low: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if re.search(r"(?<![a-z])" + re.escape(p.strip()) + r"(?![a-z])", low)]


def _sentiment(tokens: list[str]) -> tuple[float, float, list[str], list[str]]:
    pos = neg = 0.0
    pos_hits, neg_hits = [], []
    for i, t in enumerate(tokens):
        if t not in POSITIVE and t not in NEGATIVE:
            continue
        window = tokens[max(0, i - 3):i]
        weight = 1.0
        for w in window:
            weight *= INTENSIFIERS.get(w, 1.0)
        flipped = any(w in NEGATIONS for w in window)
        is_pos = (t in POSITIVE) != flipped
        label = ("not " if flipped else "") + t
        if is_pos:
            pos += weight
            pos_hits.append(label)
        else:
            neg += weight
            neg_hits.append(label)
    return pos, neg, pos_hits, neg_hits


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 1]


def analyze_tone(text: str) -> ToneReport:
    low = text.lower()
    tokens = _tokens(text)
    words = [t for t in tokens if t.isalpha()]
    n = max(len(words), 1)

    pos, neg, pos_hits, neg_hits = _sentiment(tokens)
    rude_words = [t for t in tokens if t in INSULTS]
    neg += 1.5 * len(rude_words)
    neg_hits += rude_words
    total = pos + neg
    score = 0.0 if total == 0 else (pos - neg) / total
    density = total / n
    if total == 0 or density < 0.015:
        sentiment = "neutral"
    elif pos >= 1 and neg >= 1 and pos + neg >= 2.5 and abs(score) < 0.35:
        sentiment = "mixed"
    elif score > 0.15:
        sentiment = "positive"
    elif score < -0.15:
        sentiment = "negative"
    else:
        sentiment = "mixed" if pos and neg else "neutral"

    slang = _count_phrases(low + " ", SLANG)
    formal = _count_phrases(low, FORMAL)
    contractions = len(re.findall(r"[a-z]'(?:s|re|ve|ll|d|m|t)\b", low))
    exclaim = text.count("!")
    emoji = len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", text))
    avg_len = sum(len(w) for w in words) / n
    first_second = sum(1 for w in words if w in {"i", "you", "we", "me", "my", "your"}) / n
    f = 0.5 + 0.06 * (avg_len - 4.5) + 0.08 * len(formal) - 0.10 * len(slang) - 0.04 * contractions \
        - 0.05 * min(exclaim, 4) - 0.08 * emoji - 0.3 * max(first_second - 0.08, 0)
    f = min(max(f, 0.0), 1.0)
    formality = "formal" if f >= 0.5 else "informal"

    polite = _count_phrases(low, POLITE)
    rude = _count_phrases(low, IMPOLITE)
    caps = [w for w in re.findall(r"\b[A-Z]{3,}\b", text) if w not in {"USA", "AQI", "API", "CEO", "NLP", "PDF"}]
    aggression = min(1.0, 0.25 * len(rude) + 0.15 * len(caps) + 0.1 * max(exclaim - 1, 0)
                     + 0.2 * len(re.findall(r"!{2,}", text)))
    politeness = "impolite" if (rude or aggression >= 0.5) and len(polite) <= len(rude) else \
        ("polite" if polite else "neutral")

    hedges = _count_phrases(low, HEDGES)
    boosters = _count_phrases(low, BOOSTERS)
    if len(hedges) >= len(boosters) + 2:
        confidence = "hedging"
    elif len(boosters) >= len(hedges) + 2:
        confidence = "assertive"
    else:
        confidence = "balanced"

    emotions = {k: sum(1 for w in words if w in set(v)) for k, v in EMOTIONS.items()}
    emotions = {k: v for k, v in emotions.items() if v}

    overall = [sentiment, formality]
    if politeness != "neutral":
        overall.append(politeness)
    if aggression >= 0.5:
        overall.append("aggressive")
    if confidence != "balanced":
        overall.append(confidence)
    if emotions:
        overall.append(Counter(emotions).most_common(1)[0][0])

    sentences = []
    for s in sentence_split(text):
        sp, sn, _, _ = _sentiment(_tokens(s))
        lab = "neutral" if sp + sn == 0 else ("positive" if sp > sn else "negative" if sn > sp else "mixed")
        sentences.append({"sentence": s, "sentiment": lab})

    evidence = {"positive_words": pos_hits[:10], "negative_words": neg_hits[:10], "slang": slang[:8],
                "formal_markers": formal[:8], "polite_markers": polite[:8], "impolite_markers": rude[:8],
                "hedges": hedges[:8], "boosters": boosters[:8], "all_caps": caps[:8]}
    return ToneReport(sentiment, round(score, 3), formality, round(f, 3), politeness, confidence,
                      round(aggression, 2), emotions, overall, evidence, sentences)


def tone_shifts(report: ToneReport) -> list[str]:
    """Places where the sentiment flips between consecutive sentences."""
    out = []
    labs = [s["sentiment"] for s in report.sentences]
    for i in range(1, len(labs)):
        a, b = labs[i - 1], labs[i]
        if {a, b} == {"positive", "negative"}:
            out.append(f"Sentence {i}→{i + 1}: {a} → {b}")
    return out
