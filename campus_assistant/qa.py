"""Question answering over processed announcements.

Retrieval (TF-IDF, word + char n-grams) + structured filtering on extracted fields
(course code, exam type, date range, category) + answer generation that is always
grounded in source announcements. Two answer generators:

* "rules" - template answers built from the structured fields (default, offline)
* "llm"   - Claude answers from the retrieved announcements only (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .classification import CATEGORY_EN, CATEGORY_TR
from .extraction import COURSE_STOP, RuleExtractor
from .pipeline import KIND_LABEL, detect_exam_kind
from .textutils import EN_WEEKDAYS, TR_WEEKDAYS, detect_lang, fold, format_date, tokenize

CAT_CUES = {  # folded regexes
    "exam": r"\bsinav|\bvize\b|\bfinal\b|\bquiz|\bbutunleme|\bexam|\bmidterm",
    "event": r"etkinlik|seminer|soylesi|konser|atolye|workshop|\bevent|panel|turnuva|sergi|kulup|festival",
    "scholarship": r"\bburs|scholarship|erasmus|tubitak|degisim program|hareketlilik|destek",
    "cancellation": r"iptal|yapilmayacak|yapilmiyor|cancel|no lecture|ders var mi",
    "room_change": r"derslik|hangi sinifta|room change|which room|ders nerede|dersi nerede",
    "registration": r"\bkayit|ekleme|birakma|akademik takvim|\bharc|registration|add-drop|tuition|yandal|"
                    r"cift anadal|yatay gecis|muafiyet|mezuniyet|danisman",
}
WHERE_CUES = r"\bnerede|hangi salon|hangi sinif|hangi derslik|\bwhere\b|which room|\byer\b|salonda"
WHEN_CUES = r"ne zaman|saat kac|hangi gun|\bwhen\b|what time|tarih|deadline|son gun"

NO_ANSWER = {"tr": "Bu soruyla ilgili bir duyuru bulamadım.",
             "en": "I couldn't find an announcement about that."}


@dataclass
class ParsedQuestion:
    text: str
    lang: str
    course: str | None = None
    exam_kind: str | None = None
    categories: list = field(default_factory=list)
    date_range: tuple | None = None
    range_label: str | None = None
    ask: str = "info"  # when | where | list | info

    def to_dict(self):
        d = dict(self.__dict__)
        if self.date_range:
            d["date_range"] = [x.isoformat() for x in self.date_range]
        return d


class QuestionParser:
    RE_COURSE = re.compile(r"\b([A-Za-zÇĞİÖŞÜçğıöşü]{2,4})\s?(\d{3})(?!\d)")

    def __init__(self):
        self.extractor = RuleExtractor()

    def parse(self, q: str, today: date) -> ParsedQuestion:
        lang = detect_lang(q)
        p = ParsedQuestion(text=q, lang=lang)
        t = fold(q)
        for m in self.RE_COURSE.finditer(q):
            pre = m.group(1).upper().replace("İ", "I") if m.group(1).isascii() else m.group(1).upper()
            if pre not in COURSE_STOP:
                p.course = f"{pre} {m.group(2)}"
                break
        p.exam_kind = detect_exam_kind(q)
        p.categories = [c for c, rx in CAT_CUES.items() if re.search(rx, t)]
        p.date_range, p.range_label = self.parse_range(q, t, today)
        if re.search(WHERE_CUES, t):
            p.ask = "where"
        elif re.search(WHEN_CUES, t):
            p.ask = "when"
        if p.date_range and not p.course:
            p.ask = "list"
        return p

    def parse_range(self, q, t, today):
        week_start = today - timedelta(days=today.weekday())
        if re.search(r"\bbugun|\btoday", t):
            return (today, today), "today"
        if re.search(r"\byarin|\btomorrow", t):
            d = today + timedelta(days=1)
            return (d, d), "tomorrow"
        if re.search(r"gelecek hafta|haftaya|onumuzdeki hafta|next week", t):
            s = week_start + timedelta(days=7)
            return (s, s + timedelta(days=6)), "next_week"
        if re.search(r"bu hafta|this week|hafta ici", t):
            return (week_start, week_start + timedelta(days=6)), "this_week"
        if re.search(r"bu ay|this month", t):
            s = today.replace(day=1)
            e = (s + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            return (s, e), "this_month"
        if re.search(r"gelecek ay|next month", t):
            s = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
            e = (s + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            return (s, e), "next_month"
        ents = [e for e in self.extractor.extract(q, today) if e.label == "DATE"]
        if ents:
            d = date.fromisoformat(ents[0].norm)
            return (d, d), "date"
        for names in (TR_WEEKDAYS, EN_WEEKDAYS):
            for i, n in enumerate(names):
                if re.search(rf"\b{fold(n)}\b", t):
                    d = today + timedelta(days=(i - today.weekday()) % 7)
                    return (d, d), "date"
        return None, None


class Retriever:
    def __init__(self, records):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        self.records = records
        docs = [self._doc(r) for r in records]
        self.vec = FeatureUnion([
            ("w", TfidfVectorizer(preprocessor=fold, ngram_range=(1, 2), sublinear_tf=True)),
            ("c", TfidfVectorizer(preprocessor=fold, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self.matrix = self.vec.fit_transform(docs) if docs else None

    @staticmethod
    def _doc(r):
        f = r.get("fields", {})
        return " ".join([r.get("title", ""), r.get("title", ""), f.get("subject", ""), r.get("text", "")])

    @staticmethod
    def subject_overlap(rec, q_tokens):
        subj = [t for t in tokenize(rec.get("fields", {}).get("subject", "")) if len(t) > 2]
        if not subj:
            return 0.0
        return sum(1 for t in subj if t in q_tokens) / len(subj)

    def scores(self, query: str):
        if self.matrix is None:
            return []
        qv = self.vec.transform([query])
        return (self.matrix @ qv.T).toarray().ravel() / 2.0  # two L2-normalized blocks -> [0, 1]


class QAEngine:
    def __init__(self, records, today: date | None = None, llm=None):
        self.records = list(records)
        self.today = today or date.today()
        self.parser = QuestionParser()
        self.llm = llm
        self.reindex()

    def reindex(self):
        self.retriever = Retriever(self.records)
        self.by_id = {r["id"]: r for r in self.records}

    def add(self, rec):
        self.records.append(rec)
        self.reindex()

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, p: ParsedQuestion, k: int = 8):
        sims = self.retriever.scores(p.text)
        q_tokens = set(tokenize(p.text))
        scored = []
        for r, sim in zip(self.records, sims):
            f = r.get("fields", {})
            if p.course and f.get("course") != p.course:
                continue
            s = float(sim) + 0.4 * self.retriever.subject_overlap(r, q_tokens)
            if r["category"] in p.categories:
                s += 0.15
            if p.course and p.exam_kind and r["category"] in ("exam", "room_change"):
                s += 0.3 if f.get("exam_kind") == p.exam_kind else -0.2
            if p.date_range:
                ds = [date.fromisoformat(x) for x in f.get("dates", [])]
                if any(p.date_range[0] <= d <= p.date_range[1] for d in ds):
                    s += 0.3
                elif p.ask == "list":
                    continue
            scored.append((s, r))
        scored.sort(key=lambda x: (x[0], x[1]["published"]), reverse=True)
        return scored[:k] if k else scored

    # ------------------------------------------------------------------ answering
    def answer(self, question: str, mode: str = "rules") -> dict:
        p = self.parser.parse(question, self.today)
        if p.ask == "list":
            res = self._answer_list(p)
        else:
            res = self._answer_single(p)
        res["parsed"] = p.to_dict()
        res["mode"] = "rules"
        if mode == "llm" and self.llm is not None:
            llm_text = self._llm_answer(p, res)
            if llm_text:
                res["rule_answer"], res["answer"], res["mode"] = res["answer"], llm_text, "llm"
        return res

    def _src(self, r):
        f = r.get("fields", {})
        return {"id": r["id"], "title": r["title"], "category": r["category"], "published": r["published"],
                "date": f.get("date"), "source": r.get("source")}

    def _fmt_d(self, iso, lang):
        return format_date(date.fromisoformat(iso), lang) if iso else ""

    def _past(self, iso, lang):
        if iso and date.fromisoformat(iso) < self.today:
            return " (date has passed)" if lang == "en" else " (tarihi geçti)"
        return ""

    def _latest(self, recs):
        return max(recs, key=lambda r: r["published"])

    def _room_change_for(self, exam_rec):
        """A later room-change announcement about the same exam (course + kind or date)."""
        f = exam_rec["fields"]
        best = None
        for r in self.records:
            g = r.get("fields", {})
            if r["category"] != "room_change" or g.get("course") != f.get("course"):
                continue
            same = (g.get("exam_kind") and g.get("exam_kind") == f.get("exam_kind")) or \
                   (g.get("date") and g.get("date") == f.get("date") and re.search(r"(?i)sınav|exam|quiz", r["text"]))
            if same and (best is None or r["published"] > best["published"]):
                best = r
        return best

    def _exam_view(self, recs):
        """Collapse several announcements about the same exam into its current facts."""
        groups = {}
        for r in recs:
            f = r["fields"]
            groups.setdefault((f.get("course"), f.get("exam_kind")), []).append(r)
        views = []
        for (course, kind), rs in groups.items():
            base = self._latest(rs)
            f = dict(base["fields"])
            sources = list(rs)
            rc = self._room_change_for(base)
            if rc:
                f["old_location"] = f.get("location")
                f["location"] = rc["fields"].get("location")
                sources.append(rc)
            views.append((f, sources, base["lang"]))
        views.sort(key=lambda v: (v[0].get("date") or "9999", v[0].get("time") or ""))
        return views

    def _exam_line(self, f, lang, ask):
        name = f"{f.get('course', '')} {KIND_LABEL[lang][f.get('exam_kind')]}".strip()
        d = self._fmt_d(f.get("date"), lang)
        time_s = f.get("time")
        loc = f.get("location")
        if lang == "en":
            s = f"{name}: {d}" + (f", {time_s}" if time_s else "") + (f", room {loc}" if loc else "")
            if f.get("old_location") and f.get("old_location") != loc:
                s += f" (moved from {f['old_location']})"
        else:
            s = f"{name}: {d}" + (f", saat {time_s}" if time_s else "") + (f", {loc}" if loc else "")
            if f.get("old_location") and f.get("old_location") != loc:
                s += f" (salon değişti, önceki: {f['old_location']})"
        return s + self._past(f.get("date"), lang)

    def _answer_single(self, p: ParsedQuestion):
        lang = p.lang
        hits = self.retrieve(p, k=0)
        if not hits or (not p.course and hits[0][0] < 0.25):
            return {"answer": NO_ANSWER[lang], "found": False, "sources": []}
        top_score, top = hits[0]

        wants_exam = "exam" in p.categories and "registration" not in p.categories
        if p.course and wants_exam:
            exams = [r for _, r in hits if r["category"] == "exam"]
            if p.exam_kind:
                exams = [r for r in exams if r["fields"].get("exam_kind") == p.exam_kind] or exams
            if exams:
                views = self._exam_view(exams)
                lines = [self._exam_line(f, lang, p.ask) for f, _, _ in views]
                srcs = [self._src(r) for _, rs, _ in views for r in rs]
                if len(lines) == 1:
                    return {"answer": lines[0], "found": True, "sources": srcs}
                head = f"{p.course} için {len(lines)} sınav duyurusu var:" if lang == "tr" else \
                    f"There are {len(lines)} exams for {p.course}:"
                return {"answer": head + "\n" + "\n".join("• " + x for x in lines), "found": True, "sources": srcs}
        if p.course:
            prefer = [c for c in ("cancellation", "room_change") if c in p.categories]
            pool = [r for _, r in hits if r["category"] in prefer] if prefer else []
            top = self._latest(pool) if pool else top
        else:
            # same subject announced several times (reminders): use the newest one
            same = [r for s, r in hits if r["category"] == top["category"] and s >= top_score - 0.02
                    and r["fields"].get("subject") == top["fields"].get("subject")]
            top = self._latest(same) if same else top
        return {"answer": self.describe(top, lang), "found": True, "sources": [self._src(top)]}

    def describe(self, r, lang):
        f, cat = r.get("fields", {}), r["category"]
        d = self._fmt_d(f.get("date"), lang)
        d2 = self._fmt_d(f.get("date2"), lang)
        subj = f.get("subject") or r["title"]
        subj = subj[:1].upper() + subj[1:]
        course = f.get("course", "")
        time_s, loc = f.get("time"), f.get("location")
        if cat == "exam":
            return self._exam_line(f, lang, "info")
        return self._describe(r, f, cat, lang, d, d2, subj, course, time_s, loc) + self._past(f.get("date"), lang)

    def _describe(self, r, f, cat, lang, d, d2, subj, course, time_s, loc):
        if lang == "en":
            if cat == "event":
                return f"{subj}: {d}" + (f", {time_s}" if time_s else "") + (f", {loc}" if loc else "") + "."
            if cat == "scholarship":
                return f"The application deadline for {subj} is {d}." if d else r["summary"]
            if cat == "cancellation":
                s = f"Yes, the {course} class on {d} is cancelled."
                if d2:
                    s += f" Make-up class: {d2}" + (f", {time_s}" if time_s else "") + (f", {loc}" if loc else "") + "."
                return s
            if cat == "room_change":
                return f"From {d}, {course} takes place in {loc}" + \
                    (f" (instead of {f['old_location']})." if f.get("old_location") else ".")
            if cat == "registration":
                return f"{subj}: starts {d}" + (f", ends {d2}" if d2 else "") + "."
        else:
            if cat == "event":
                return f"{subj}: {d}" + (f", saat {time_s}" if time_s else "") + (f", {loc}" if loc else "") + "."
            if cat == "scholarship":
                return f"{subj} için son başvuru tarihi: {d}." if d else r["summary"]
            if cat == "cancellation":
                s = f"Evet, {course} dersi {d} günü yapılmayacak."
                if d2:
                    s += f" Telafi: {d2}" + (f", saat {time_s}" if time_s else "") + (f", {loc}" if loc else "") + "."
                return s
            if cat == "room_change":
                return f"{course} dersi {d} itibarıyla {loc} salonunda yapılacak" + \
                    (f" (önceki: {f['old_location']})." if f.get("old_location") else ".")
            if cat == "registration":
                return f"{subj}: başlangıç {d}" + (f", bitiş {d2}" if d2 else "") + "."
        return r.get("summary") or r["title"]

    def _answer_list(self, p: ParsedQuestion):
        lang = p.lang
        cats = [c for c in p.categories if c != "room_change"] or None
        hits = [r for _, r in self.retrieve(p, k=0)]
        hits = [r for r in hits if (cats is None or r["category"] in cats)
                and r["fields"].get("date") and p.date_range[0] <= date.fromisoformat(r["fields"]["date"])
                <= p.date_range[1]]
        label_tr = {"today": "Bugün", "tomorrow": "Yarın", "this_week": "Bu hafta", "next_week": "Gelecek hafta",
                    "this_month": "Bu ay", "next_month": "Gelecek ay", "date": format_date(p.date_range[0], "tr")}
        label_en = {"today": "today", "tomorrow": "tomorrow", "this_week": "this week", "next_week": "next week",
                    "this_month": "this month", "next_month": "next month",
                    "date": "on " + format_date(p.date_range[0], "en")}
        what_tr = CATEGORY_TR.get(cats[0], "duyuru").lower() if cats and len(cats) == 1 else "duyuru"
        what_en = CATEGORY_EN.get(cats[0], "announcement").lower() if cats and len(cats) == 1 else "announcement"
        if not hits:
            msg = f"{label_tr[p.range_label]} için {what_tr} kaydı bulamadım." if lang == "tr" else \
                f"I found no {what_en} announcements {label_en[p.range_label]}."
            return {"answer": msg, "found": False, "sources": []}
        lines, srcs = [], []
        exams = [r for r in hits if r["category"] == "exam"]
        for f, rs, _ in self._exam_view(exams):
            lines.append((f.get("date"), f.get("time") or "", self._exam_line(f, lang, "list")))
            srcs += [self._src(r) for r in rs]
        seen = set()
        for r in sorted((r for r in hits if r["category"] != "exam"), key=lambda r: (r["lang"] == lang, r["published"]),
                        reverse=True):
            g = r["fields"]
            key = (r["category"], g.get("date"), g.get("time"), g.get("location")) if r["category"] == "event" \
                else (r["category"], g.get("subject") or g.get("course"), g.get("date"))
            if key in seen:
                continue
            seen.add(key)
            lines.append((r["fields"].get("date"), r["fields"].get("time") or "", self.describe(r, lang)))
            srcs.append(self._src(r))
        lines.sort()
        n = len(lines)
        if lang == "tr":
            head = f"{label_tr[p.range_label]} {n} {what_tr} var:"
        else:
            head = f"Yes, {n} {what_en}{'s' if n > 1 else ''} {label_en[p.range_label]}:"
        return {"answer": head + "\n" + "\n".join("• " + x[2] for x in lines[:15]), "found": True,
                "sources": srcs}

    # ------------------------------------------------------------------ LLM grounded answer
    def _llm_answer(self, p: ParsedQuestion, rule_res: dict):
        ids = [s["id"] for s in rule_res.get("sources", [])]
        extra = [r["id"] for _, r in self.retrieve(p, k=6)]
        ctx_ids = list(dict.fromkeys(ids + extra))[:10]
        ctx = [{"id": i, "published": self.by_id[i]["published"], "category": self.by_id[i]["category"],
                "title": self.by_id[i]["title"], "text": self.by_id[i]["text"],
                "extracted": self.by_id[i]["fields"]} for i in ctx_ids]
        language = "English" if p.lang == "en" else "Turkish"
        system = ("You are a university campus announcement assistant. Answer ONLY from the announcements "
                  "provided. If they do not contain the answer, say so. When announcements conflict, the most "
                  "recently published one wins (e.g. a room change overrides the original exam room). Keep "
                  "dates, times, rooms and course codes exactly as written, and cite announcement ids in "
                  "square brackets.")
        prompt = (f"Today is {format_date(self.today, 'en')}.\n\nAnnouncements (JSON):\n"
                  f"{json.dumps(ctx, ensure_ascii=False, indent=1)}\n\n"
                  f"Question: {p.text}\nAnswer briefly in {language}.")
        return self.llm.complete(prompt, system=system, max_tokens=1024)
