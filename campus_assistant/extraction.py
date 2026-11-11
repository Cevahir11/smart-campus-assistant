"""Information extraction: dates, times, locations and course codes.

Two extractors share one interface (`extract(text, ref_date) -> list[Entity]`):

* RuleExtractor      - regex + gazetteer rules for Turkish and English (baseline, no deps)
* HybridExtractor    - rules + a fine-tuned transformer NER model (BERTurk / XLM-R),
                       loaded from models/ner if present (see training/train_ner.py)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .textutils import EN_MONTHS, TR_MONTHS, tr_lower


@dataclass
class Entity:
    label: str      # DATE | TIME | LOCATION | COURSE
    text: str       # surface string
    norm: str       # normalized value (ISO date, HH:MM, canonical room, "CENG 301")
    start: int
    end: int
    source: str = "rule"

    def to_dict(self):
        return asdict(self)


_MONTH_TR = {tr_lower(m): i + 1 for i, m in enumerate(TR_MONTHS)}
_MONTH_TR.update({"subat": 2, "mayis": 5, "agustos": 8, "eylul": 9, "kasim": 11, "aralik": 12})
_MONTH_EN = {m.lower(): i + 1 for i, m in enumerate(EN_MONTHS)}
_MONTH_EN.update({m.lower()[:3]: i + 1 for i, m in enumerate(EN_MONTHS)})
_TR_M = "|".join(sorted(TR_MONTHS + ["Subat", "Mayis", "Agustos", "Eylul", "Kasim", "Aralik"], key=len, reverse=True))
_EN_M = "|".join(sorted(list(EN_MONTHS) + [m[:3] for m in EN_MONTHS], key=len, reverse=True))

RE_NUM_DATE = re.compile(r"(?<![\d.])(\d{1,2})[./](\d{1,2})[./](\d{4})(?![\d])")
RE_TR_RANGE = re.compile(rf"(?<!\d)(\d{{1,2}})\s*[-–]\s*(\d{{1,2}})\s+({_TR_M})(?:\s+(\d{{4}}))?", re.I)
RE_TR_DATE = re.compile(rf"(?<!\d)(\d{{1,2}})\s+({_TR_M})\b(?:\s+(\d{{4}}))?", re.I)
RE_EN_MDY = re.compile(rf"\b({_EN_M})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
RE_EN_DMY = re.compile(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_EN_M})\b(?:,?\s+(\d{{4}}))?", re.I)

RE_TIME_COLON = re.compile(r"(?<![\d:.])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
RE_TIME_DOT = re.compile(r"(?i)(?:saat|saati)\s*:?\s*([01]?\d|2[0-3])\.([0-5]\d)(?![\d.]\d)")
RE_TIME_HOUR_TR = re.compile(r"(?i)(?:saat|saati)\s+([01]?\d|2[0-3])['’](?:da|de|ta|te)\b")
RE_TIME_AFTERNOON_TR = re.compile(r"(?i)öğleden sonra\s+(?:saat\s+)?([1-9]|1[01])['’](?:da|de|ta|te)\b")
RE_TIME_AMPM = re.compile(r"(?i)\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b")

RE_COURSE = re.compile(r"\b([A-ZÇĞİÖŞÜ]{2,4})\s?(\d{3})(?![\d])")
COURSE_STOP = {"TL", "SAAT", "NO", "TEL", "AM", "PM", "GPA", "TC", "ID"}

GAZETTEER = ["Konferans Salonu", "Kütüphane Seminer Salonu", "Kültür Merkezi", "Kongre Merkezi",
             "Spor Salonu", "Yemekhane", "Rektörlük Binası", "Mühendislik Amfisi", "Conference Hall",
             "Library Seminar Room", "Culture Center", "Congress Center", "Sports Hall"]
RE_LOC = [
    re.compile(r"(?<![\w-])([A-Z])-(\d{2,3})(?![\w])"),                 # B-204, Z-12
    re.compile(r"\b(Amfi|Amphitheater|Amphi)\s?(\d{1,2})\b", re.I),       # Amfi 3
    re.compile(r"\b([A-Z])\s+Blok\s+(\d{2,3})\b"),                       # D Blok 105
    re.compile(r"\b(Lab|Laboratuvar|Laboratory)\s?(\d{1,2})\b", re.I),    # Lab 2
    re.compile(r"\b(?:Derslik|Salon|Room|Classroom)\s+([A-Z]?\d{2,3})\b"),  # Derslik 204
]
RE_GAZ = re.compile("|".join(re.escape(g) for g in sorted(GAZETTEER, key=len, reverse=True)))


def _mk_date(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _infer_year(month: int, day: int, ref: date) -> int:
    """Pick the year that puts the date closest to the reference date."""
    best = None
    for y in (ref.year - 1, ref.year, ref.year + 1):
        cand = _mk_date(y, month, day)
        if cand and (best is None or abs((cand - ref).days) < abs((best - ref).days)):
            best = cand
    return best.year if best else ref.year


def _overlaps(span, taken):
    return any(span[0] < e and s < span[1] for s, e in taken)


def normalize_location(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw.strip())
    m = re.match(r"(?i)(amfi|amphitheater|amphi)\s?(\d+)", raw)
    if m:
        return f"Amfi {m.group(2)}"
    m = re.match(r"(?i)(lab|laboratuvar|laboratory)\s?(\d+)", raw)
    if m:
        return f"Lab {m.group(2)}"
    m = re.match(r"(?i)(?:derslik|salon|room|classroom)\s+(.+)", raw)
    if m:
        return m.group(1)
    return raw


class RuleExtractor:
    name = "rules"

    def extract(self, text: str, ref_date: date | None = None) -> list[Entity]:
        ref = ref_date or date.today()
        ents: list[Entity] = []
        taken: list[tuple[int, int]] = []

        def add(label, s, e, norm):
            if _overlaps((s, e), taken):
                return
            ents.append(Entity(label, text[s:e], norm, s, e))
            taken.append((s, e))

        # ---- dates (most specific first)
        for m in RE_NUM_DATE.finditer(text):
            d = _mk_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if d:
                add("DATE", m.start(), m.end(), d.isoformat())
        for m in RE_TR_RANGE.finditer(text):
            mon = _MONTH_TR.get(tr_lower(m.group(3)))
            y = int(m.group(4)) if m.group(4) else _infer_year(mon, int(m.group(2)), ref)
            d1, d2 = _mk_date(y, mon, int(m.group(1))), _mk_date(y, mon, int(m.group(2)))
            if d1 and d2 and not _overlaps((m.start(), m.end()), taken):
                ents.append(Entity("DATE", m.group(1), d1.isoformat(), m.start(1), m.end(1)))
                ents.append(Entity("DATE", text[m.start(2):m.end()], d2.isoformat(), m.start(2), m.end()))
                taken.append((m.start(), m.end()))
        for m in RE_TR_DATE.finditer(text):
            mon = _MONTH_TR.get(tr_lower(m.group(2)))
            if not mon:
                continue
            y = int(m.group(3)) if m.group(3) else _infer_year(mon, int(m.group(1)), ref)
            d = _mk_date(y, mon, int(m.group(1)))
            if d:
                add("DATE", m.start(), m.end(), d.isoformat())
        for rx, order in ((RE_EN_MDY, "md"), (RE_EN_DMY, "dm")):
            for m in rx.finditer(text):
                if order == "md":
                    mon, day = _MONTH_EN.get(m.group(1).lower()), int(m.group(2))
                else:
                    mon, day = _MONTH_EN.get(m.group(2).lower()), int(m.group(1))
                if not mon:
                    continue
                y = int(m.group(3)) if m.group(3) else _infer_year(mon, day, ref)
                d = _mk_date(y, mon, day)
                if d:
                    add("DATE", m.start(), m.end(), d.isoformat())

        # ---- times
        for m in RE_TIME_COLON.finditer(text):
            add("TIME", m.start(), m.end(), f"{int(m.group(1)):02d}:{m.group(2)}")
        for m in RE_TIME_DOT.finditer(text):
            add("TIME", m.start(1), m.end(2), f"{int(m.group(1)):02d}:{m.group(2)}")
        for m in RE_TIME_HOUR_TR.finditer(text):
            add("TIME", m.start(1), m.end(1), f"{int(m.group(1)):02d}:00")
        for m in RE_TIME_AFTERNOON_TR.finditer(text):
            add("TIME", m.start(1), m.end(1), f"{int(m.group(1)) + 12:02d}:00")
        for m in RE_TIME_AMPM.finditer(text):
            h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
            add("TIME", m.start(), m.end(), f"{h:02d}:{m.group(2) or '00'}")

        # ---- course codes
        for m in RE_COURSE.finditer(text):
            if m.group(1) in COURSE_STOP:
                continue
            add("COURSE", m.start(), m.end(), f"{m.group(1)} {m.group(2)}")

        # ---- locations
        for m in RE_GAZ.finditer(text):
            add("LOCATION", m.start(), m.end(), m.group(0))
        for rx in RE_LOC:
            for m in rx.finditer(text):
                add("LOCATION", m.start(), m.end(), normalize_location(m.group(0)))

        ents.sort(key=lambda e: e.start)
        return ents


class HybridExtractor(RuleExtractor):
    """Rules + transformer token-classification model. Model predictions add entities
    the rules missed; normalization still goes through the rule normalizers."""
    name = "hybrid"

    def __init__(self, model_dir: str | Path):
        from transformers import pipeline  # optional dependency
        self.ner = pipeline("token-classification", model=str(model_dir), aggregation_strategy="simple")

    def extract(self, text, ref_date=None):
        ents = super().extract(text, ref_date)
        taken = [(e.start, e.end) for e in ents]
        for p in self.ner(text):
            s, e, label = int(p["start"]), int(p["end"]), p["entity_group"]
            if _overlaps((s, e), taken):
                continue
            sub = super().extract(text[s:e], ref_date)
            norm = sub[0].norm if sub and sub[0].label == label else text[s:e].strip()
            if label == "LOCATION":
                norm = normalize_location(text[s:e])
            ents.append(Entity(label, text[s:e], norm, s, e, source="model"))
            taken.append((s, e))
        ents.sort(key=lambda e: e.start)
        return ents


def load_extractor(model_dir: str | Path | None = None):
    model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[1] / "models" / "ner"
    if (model_dir / "config.json").exists():
        try:
            return HybridExtractor(model_dir)
        except Exception as exc:  # transformers not installed etc.
            print(f"[extraction] NER model not loaded ({exc}); using rules only")
    return RuleExtractor()
