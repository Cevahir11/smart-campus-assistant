"""Shared text helpers: Turkish-aware lowercasing, sentence splitting, tokenizing,
and date formatting in Turkish / English."""
from __future__ import annotations

import re
import unicodedata
from datetime import date

TR_MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
             "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
TR_WEEKDAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
EN_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def tr_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless I rules."""
    return text.translate(_TR_LOWER).lower()


def fold(text: str) -> str:
    """Lowercase and strip diacritics (ş->s, ı->i ...) for robust keyword matching."""
    text = tr_lower(text).replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    """Basic cleaning for scraped announcements: strip HTML leftovers and whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        parts.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return parts


_TOKEN = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [tr_lower(t) for t in _TOKEN.findall(text)]


def format_date(d: date, lang: str = "tr", weekday: bool = True) -> str:
    if lang == "en":
        s = f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}"
        return f"{EN_WEEKDAYS[d.weekday()]}, {s}" if weekday else s
    s = f"{d.day} {TR_MONTHS[d.month - 1]} {d.year}"
    return f"{s} {TR_WEEKDAYS[d.weekday()]}" if weekday else s


def format_date_short(d: date, lang: str = "tr") -> str:
    """The part an answer must always contain: '14 Kasım' / 'November 14'."""
    if lang == "en":
        return f"{EN_MONTHS[d.month - 1]} {d.day}"
    return f"{d.day} {TR_MONTHS[d.month - 1]}"


_EN_HINTS = {"the", "is", "are", "when", "where", "what", "which", "any", "this", "next",
             "exam", "exams", "event", "events", "there", "week", "scholarship", "deadline",
             "class", "cancelled", "canceled", "room", "registration", "today", "tomorrow"}
_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")


def detect_lang(text: str) -> str:
    """Very small heuristic language id (tr / en), good enough for routing answers."""
    if any(ch in _TR_CHARS for ch in text):
        return "tr"
    toks = tokenize(text)
    if not toks:
        return "tr"
    en_hits = sum(1 for t in toks if t in _EN_HINTS)
    return "en" if en_hits >= max(1, len(toks) // 4) else "tr"
