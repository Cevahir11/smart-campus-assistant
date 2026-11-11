"""Application service shared by the web app and the CLI: loads announcements,
runs the NLP pipeline over them and answers questions."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from .classification import CATEGORY_EN, CATEGORY_TR
from .llm import ClaudeLLM
from .pipeline import AnnouncementProcessor, dataset_reference_date, read_jsonl
from .qa import QAEngine
from .textutils import fold

ROOT = Path(__file__).resolve().parents[1]


def resolve_today(data_dir: Path) -> date:
    """CAMPUS_TODAY=YYYY-MM-DD | system ; default: the dataset's reference date (demo mode)."""
    v = os.environ.get("CAMPUS_TODAY", "").strip().lower()
    if v == "system":
        return date.today()
    if v:
        return date.fromisoformat(v)
    return dataset_reference_date(data_dir)


class AssistantService:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or ROOT / "data")
        self.user_file = self.data_dir / "user_announcements.jsonl"
        base = read_jsonl(self.data_dir / "announcements.jsonl")
        # CAMPUS_FEED = synthetic (default) | gtu (real GTÜ announcements from data/gtu/*.jsonl) | all
        self.feed = os.environ.get("CAMPUS_FEED", "synthetic").strip().lower()
        gtu = [r for p in sorted((self.data_dir / "gtu").glob("*.jsonl")) for r in read_jsonl(p)] \
            if self.feed in ("gtu", "all") else []
        if self.feed == "gtu" and not os.environ.get("CAMPUS_TODAY"):
            self.today = date.today()  # real announcements -> real calendar
        else:
            self.today = resolve_today(self.data_dir)
        self.processor = AnnouncementProcessor(train_records=base)  # classifier still trained on labelled data
        shown = gtu if self.feed == "gtu" else base + gtu
        raw = shown + (read_jsonl(self.user_file) if self.user_file.exists() else [])
        self.records = [self.processor.process(r) for r in raw]
        self.llm = ClaudeLLM() if ClaudeLLM.available() else None
        self.qa = QAEngine(self.records, today=self.today, llm=self.llm)

    # -------------------------------------------------------------- queries
    def ask(self, question: str, mode: str = "rules") -> dict:
        res = self.qa.answer(question, mode=mode if self.llm else "rules")
        res["llm_available"] = self.llm is not None
        return res

    def list(self, category: str | None = None, q: str | None = None, upcoming: bool = False, limit: int = 200):
        recs = self.records
        if category:
            recs = [r for r in recs if r["category"] == category]
        if q:
            fq = fold(q)
            recs = [r for r in recs if fq in fold(f"{r['title']} {r['text']}")]
        if upcoming:
            recs = [r for r in recs if r["fields"].get("date", "0000") >= self.today.isoformat()]
            recs = sorted(recs, key=lambda r: (r["fields"].get("date"), r["fields"].get("time") or ""))
        else:
            recs = sorted(recs, key=lambda r: (r["published"], r["id"]), reverse=True)
        return recs[:limit]

    def analyze(self, title: str, text: str, published: str | None = None) -> dict:
        return self.processor.process({"id": "preview", "title": title, "text": text,
                                       "published": published or self.today.isoformat(), "source": "user"})

    def add(self, title: str, text: str, source: str = "user", published: str | None = None) -> dict:
        import json
        raw = {"id": f"user-{len(self.records) + 1:04d}", "title": title, "text": text, "source": source,
               "published": published or self.today.isoformat()}
        self.user_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.user_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
        rec = self.processor.process(raw)
        self.records.append(rec)
        self.qa.add(rec)
        return rec

    def stats(self):
        counts = {}
        for r in self.records:
            counts[r["category"]] = counts.get(r["category"], 0) + 1
        return {"total": len(self.records), "by_category": counts, "today": self.today.isoformat(),
                "llm_available": self.llm is not None, "classifier": getattr(self.processor.classifier, "name", "?"),
                "extractor": getattr(self.processor.extractor, "name", "?"),
                "categories_tr": CATEGORY_TR, "categories_en": CATEGORY_EN}
