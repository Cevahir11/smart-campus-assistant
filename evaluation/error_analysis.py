"""Group the baseline errors dumped by evaluation/midproject.py into categories.

    python evaluation/error_analysis.py    -> reports/midproject_errors.csv + counts on stdout

Categories were defined by reading all errors once (Nov 11, 2026); the rules below reproduce
that manual grouping so the counts can be re-derived.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campus_assistant.classification import KEYWORDS  # noqa: E402
from campus_assistant.textutils import fold  # noqa: E402

CATS = {
    "E1": "Topic beats action: exam room change -> exam",
    "E2": "Shared cue 'basvuru/applications': registration -> scholarship",
    "E3": "Registration about an exam (exam applications) -> exam",
    "E4": "No keyword fires -> fallback class",
    "E5": "Event text mentions exams ('sinav haftasi oncesi') -> exam",
    "E6": "Room name contains event cue ('Seminer Salonu') -> event",
    "E7": "TF-IDF: unseen wording outside templates",
    "X1": "Extraction: date range without year resolved to wrong year",
    "X2": "Extraction: unseen free-form location",
    "X3": "Extraction: dotted time '17.30'da' missed",
}


def categorize(e):
    if e["model"] == "tfidf_logreg":
        return "E7"
    if e["model"] == "rule_extractor":
        if "DATE" in e["gold"] or "DATE" in e["pred"]:
            return "X1"
        return "X2" if "LOCATION" in e["gold"] else "X3"
    t = fold(e["title"] + ". " + e["text"])
    hits = {c for c, kws in KEYWORDS.items() if any(k in t for k in kws)}
    if not hits:
        return "E4"
    if e["gold"] == "room_change":
        return "E1" if e["pred"] == "exam" else "E6"
    if e["gold"] == "event":
        return "E5"
    if e["gold"] == "registration":
        return "E2" if e["pred"] == "scholarship" else "E3"
    return "E4"


def main():
    res = json.loads((ROOT / "reports" / "midproject_results.json").read_text(encoding="utf-8"))
    rows = []
    for e in res["errors"]:
        rows.append({"category": categorize(e), **{k: e[k] for k in ("model", "set", "id", "lang", "gold", "pred",
                                                                      "title", "text")}})
    with open(ROOT / "reports" / "midproject_errors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cnt = Counter(r["category"] for r in rows)
    for k in CATS:
        print(f"{k} {cnt[k]:3d}  {CATS[k]}")
    print("classification errors:", sum(v for k, v in cnt.items() if k.startswith("E")),
          "| extraction errors:", sum(v for k, v in cnt.items() if k.startswith("X")))


if __name__ == "__main__":
    main()
