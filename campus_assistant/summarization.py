"""One-sentence announcement summaries.

* LeadSummarizer        - first sentence (classic news baseline)
* CentroidSummarizer    - extractive: sentence closest to the TF-IDF centroid of the document
* StructuredSummarizer  - template filled from the predicted category + extracted entities
* Seq2SeqSummarizer     - mT5 (e.g. csebuetnlp/mT5_multilingual_XLSum or your fine-tuned model)
* LLMSummarizer         - prompted Claude model (needs ANTHROPIC_API_KEY), falls back to Structured
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import date

from .textutils import format_date_short, split_sentences, tokenize


class LeadSummarizer:
    name = "lead_1"

    def summarize(self, rec: dict) -> str:
        sents = split_sentences(rec["text"])
        return sents[0] if sents else rec.get("title", "")


class CentroidSummarizer:
    name = "centroid_extractive"

    def summarize(self, rec: dict) -> str:
        sents = split_sentences(rec["text"])
        if not sents:
            return rec.get("title", "")
        toks = [tokenize(s) for s in sents]
        df = Counter(t for ts in toks for t in set(ts))
        n = len(sents)

        def vec(ts):
            tf = Counter(ts)
            return {t: c * math.log(1 + n / df[t]) for t, c in tf.items()}

        vecs = [vec(ts) for ts in toks]
        centroid = Counter()
        for v in vecs:
            centroid.update(v)
        title_toks = set(tokenize(rec.get("title", "")))

        def score(i):
            v = vecs[i]
            dot = sum(w * centroid[t] for t, w in v.items())
            norm = math.sqrt(sum(w * w for w in v.values())) * math.sqrt(sum(w * w for w in centroid.values()))
            overlap = len(title_toks & set(toks[i])) / (len(title_toks) or 1)
            return (dot / norm if norm else 0) + 0.5 * overlap - 0.02 * i

        return sents[max(range(n), key=score)]


def tr_locative(loc: str) -> str:
    """'Konferans Salonu' -> "Konferans Salonu'nda", 'B-204' -> 'B-204 salonunda'."""
    low = loc.lower()
    if low.endswith(("salonu", "binası", "amfisi")):
        return f"{loc}'nda"
    if low.endswith("merkezi"):
        return f"{loc}'nde"
    if low.startswith("lab"):
        return f"{loc} laboratuvarında"
    return f"{loc} salonunda"


TEMPLATES = {
    "tr": {
        "exam": "{course}{subject} {date}{time}{loc} yapılacak.",
        "event": "{subject} {date}{time}{loc} gerçekleşecek.",
        "scholarship": "{subject} için son başvuru tarihi {date}.",
        "cancellation": "{course} dersi {date} tarihinde yapılmayacak.",
        "room_change": "{course} dersi/sınavı {date} tarihinden itibaren {loc} yapılacak.",
        "registration": "{subject} {date} tarihinde başlıyor.",
    },
    "en": {
        "exam": "{course}{subject} will be held on {date}{time}{loc}.",
        "event": "{subject} takes place on {date}{time}{loc}.",
        "scholarship": "The deadline for {subject} is {date}.",
        "cancellation": "{course} class on {date} is cancelled.",
        "room_change": "{course} moves to {loc_plain} from {date}.",
        "registration": "{subject} starts on {date}.",
    },
}


class StructuredSummarizer:
    """Fill a category template from the structured record built by the pipeline."""
    name = "structured_template"

    def summarize(self, rec: dict) -> str:
        cat, lang = rec.get("category"), rec.get("lang", "tr")
        f = rec.get("fields", {})
        if cat not in TEMPLATES["tr"] or not f.get("date"):
            return LeadSummarizer().summarize(rec)
        d = date.fromisoformat(f["date"])
        subject = f.get("subject") or rec.get("title", "")
        course = f.get("course", "")
        loc = f.get("location", "")
        if lang == "en":
            vals = dict(course=course + " " if course and cat == "exam" else course,
                        subject=f.get("exam_label", "exam") if cat == "exam" else subject,
                        date=format_date_short(d, "en"),
                        time=f" at {f['time']}" if f.get("time") else "",
                        loc=f" in {loc}" if loc else "", loc_plain=loc or "a new room")
        else:
            vals = dict(course=course + " " if course and cat == "exam" else course,
                        subject=f.get("exam_label", "sınavı") if cat == "exam" else subject,
                        date=format_date_short(d, "tr") + (" tarihinde" if cat in ("exam", "event") else ""),
                        time=f" saat {f['time']}'da" if f.get("time") else "",
                        loc=" " + tr_locative(loc) if loc else "", loc_plain=loc or "yeni")
        if cat in ("cancellation", "room_change") and not course:
            return LeadSummarizer().summarize(rec)
        text = TEMPLATES[lang][cat].format(**vals)
        return " ".join(text.split())


class Seq2SeqSummarizer:
    name = "mt5"

    def __init__(self, model_name: str = "csebuetnlp/mT5_multilingual_XLSum"):
        from transformers import pipeline
        self.pipe = pipeline("summarization", model=model_name)

    def summarize(self, rec: dict) -> str:
        out = self.pipe(rec["text"], max_length=48, min_length=8, no_repeat_ngram_size=3)
        return out[0]["summary_text"]


class LLMSummarizer:
    name = "llm"

    def __init__(self, llm=None):
        from .llm import ClaudeLLM
        self.llm = llm or ClaudeLLM()
        self.fallback = StructuredSummarizer()

    def summarize(self, rec: dict) -> str:
        lang = "English" if rec.get("lang") == "en" else "Turkish"
        prompt = (f"Summarize this university announcement in ONE short sentence in {lang}. Keep every "
                  f"date, time, room and course code exactly as written. Output only the sentence.\n\n"
                  f"Title: {rec.get('title', '')}\n{rec['text']}")
        out = self.llm.complete(prompt, max_tokens=200)
        return out.strip() if out else self.fallback.summarize(rec)
