"""End-to-end processing: raw announcement -> structured, searchable record."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from .classification import doc_text, load_classifier
from .extraction import load_extractor
from .summarization import StructuredSummarizer
from .textutils import clean_text, detect_lang, fold

ROOT = Path(__file__).resolve().parents[1]

EXAM_KINDS = [  # (folded pattern, kind)
    (r"\bara sinav|\bvize\b|\bmidterm", "midterm"),
    (r"\bfinal\b", "final"),
    (r"\bquiz\b", "quiz"),
    (r"\bbutunleme|\bresit", "resit"),
    (r"\bmazeret sinav|\bmake-?up exam", "makeup"),
]
KIND_LABEL = {"tr": {"midterm": "ara sınavı", "final": "final sınavı", "quiz": "quiz",
                     "resit": "bütünleme sınavı", "makeup": "mazeret sınavı", None: "sınavı"},
              "en": {"midterm": "midterm exam", "final": "final exam", "quiz": "quiz",
                     "resit": "resit exam", "makeup": "make-up exam", None: "exam"}}

DEADLINE_CUES = re.compile(r"(?i)(son başvuru|son gün|en geç|deadline|last day)[^.]{0,40}$")
DEADLINE_AFTER = re.compile(r"(?i)^[^.]{0,25}(kadar|sona er|tarihine kadar)")

TITLE_BOILERPLATE = [
    r"(?i)\s*etkinliğine davetlisiniz$", r"(?i)^söyleşi:\s*", r"(?i)^etkinlik duyurusu:\s*",
    r"(?i)\s*başvuruları başladı$", r"(?i)\s*duyurusu$", r"(?i)\s*-\s*son gün yaklaşıyor$",
    r"(?i)\s*bilgilendirme toplantısı$", r"(?i)\s*applications are open$", r"(?i)\s*deadline reminder$",
    r"(?i)^join us:\s*", r"(?i)\s*hakkında$", r"(?i)\s*takvimi$", r"(?i)^hatırlatma:\s*",
    r"(?i)^öğrenci işleri:\s*", r"(?i)\s*schedule$", r"(?i)^reminder:\s*", r"^.*? - ",
]


def detect_exam_kind(text: str):
    t = fold(text)
    for pat, kind in EXAM_KINDS:
        if re.search(pat, t):
            return kind
    return None


def clean_subject(title: str) -> str:
    s = title.strip()
    for pat in TITLE_BOILERPLATE:
        s = re.sub(pat, "", s).strip()
    return s or title.strip()


class AnnouncementProcessor:
    def __init__(self, classifier=None, extractor=None, summarizer=None, train_records=None):
        self.classifier = classifier or load_classifier(train_records=train_records)
        self.extractor = extractor or load_extractor()
        self.summarizer = summarizer or StructuredSummarizer()

    def process(self, raw: dict, ref_date: date | None = None) -> dict:
        title = clean_text(raw.get("title", ""))
        text = clean_text(raw.get("text", ""))
        published = raw.get("published") or (ref_date or date.today()).isoformat()
        ref = date.fromisoformat(published)
        rec = {"id": raw.get("id"), "source": raw.get("source", "unknown"), "published": published,
               "title": title, "text": text, "lang": raw.get("lang") or detect_lang(f"{title} {text}"),
               "url": raw.get("url")}

        probs = self.classifier.predict_proba([doc_text(rec)])[0]
        cat = max(probs, key=probs.get)
        rec["category"] = cat
        rec["confidence"] = round(float(probs[cat]), 3)

        ents = self.extractor.extract(text, ref)
        title_ents = self.extractor.extract(title, ref)
        rec["entities"] = [e.to_dict() for e in ents]
        rec["fields"] = self.build_fields(rec, ents, title_ents)
        rec["summary"] = self.summarizer.summarize(rec)
        return rec

    # ------------------------------------------------------------------ slot filling
    def build_fields(self, rec, ents, title_ents):
        cat, text, lang = rec["category"], rec["text"], rec["lang"]
        by = {}
        for e in ents:
            by.setdefault(e.label, []).append(e)
        f: dict = {}
        courses = by.get("COURSE") or [e for e in title_ents if e.label == "COURSE"]
        if courses:
            f["course"] = courses[0].norm
        dates = by.get("DATE", [])
        if dates:
            f["dates"] = [d.norm for d in dates]
            main = dates[0]
            if cat in ("scholarship", "registration"):
                for d in dates:
                    if DEADLINE_CUES.search(text[max(0, d.start - 60):d.start]) or \
                            DEADLINE_AFTER.search(text[d.end:d.end + 40]):
                        main = d
                        break
            f["date"] = main.norm
            later = [d for d in dates if d is not main]
            if later:
                f["date2"] = later[0].norm
        times = by.get("TIME", [])
        if times:
            f["time"] = times[0].norm
            if len(times) > 1:
                f["end_time"] = times[1].norm
        locs = by.get("LOCATION", [])
        if locs:
            if cat == "room_change" and len(locs) > 1:
                new = locs[-1]
                for i, l in enumerate(locs):
                    if re.match(r"(?i)\s*(instead of)", text[l.end:l.end + 15]):
                        new = l
                    if re.match(r"(?i)\s*(yerine|dersliğinden|olarak değil)", text[l.end:l.end + 20]):
                        new = locs[i + 1] if i + 1 < len(locs) else l
                    if re.search(r"(?i)\bfrom\s*$", text[max(0, l.start - 6):l.start]):
                        new = locs[i + 1] if i + 1 < len(locs) else l
                f["location"] = new.norm
                f["old_location"] = next(l.norm for l in locs if l is not new)
            else:
                f["location"] = locs[0].norm
        if cat in ("exam", "room_change"):
            kind = detect_exam_kind(f"{rec['title']} {text}" if cat == "exam" else text)
            if kind or cat == "exam":
                f["exam_kind"] = kind
                f["exam_label"] = KIND_LABEL[lang][kind]
        if cat in ("event", "scholarship", "registration"):
            f["subject"] = clean_subject(rec["title"])
        return f


# ---------------------------------------------------------------------- data helpers
def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def dataset_reference_date(data_dir=None) -> date:
    meta = Path(data_dir or ROOT / "data") / "meta.json"
    if meta.exists():
        return date.fromisoformat(json.loads(meta.read_text())["base_date"])
    return date.today()
