"""Precision / recall / F1 per class (and macro) on the frozen mid-project split.

    python evaluation/per_class_report.py      -> reports/midproject_per_class.md (+ stdout)
Uses the same frozen split (data/splits/midproject_split.json) and training data as evaluation/midproject.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campus_assistant.classification import CATEGORIES, KeywordClassifier, TfidfClassifier, doc_text  # noqa: E402
from campus_assistant.metrics import classification_report  # noqa: E402
from campus_assistant.pipeline import read_jsonl  # noqa: E402
from evaluation.midproject import make_or_load_split  # noqa: E402


def main():
    recs = read_jsonl(ROOT / "data" / "announcements.jsonl")
    ch = read_jsonl(ROOT / "data" / "challenge_set.jsonl")
    sp = make_or_load_split(recs)
    L = ["# Per-class precision / recall / F1 (frozen mid-project split)", ""]
    for m in [KeywordClassifier(), TfidfClassifier("logreg")]:
        m.fit([doc_text(r) for r in sp["train"]], [r["category"] for r in sp["train"]])
        for name, rows in [("Test A (synthetic, n=84)", sp["test"]), ("Test B (hand-written, n=24)", ch)]:
            gold = [r["category"] for r in rows]
            rep = classification_report(gold, m.predict([doc_text(r) for r in rows]), CATEGORIES)
            pc = rep["per_class"]
            mp = sum(v["precision"] for v in pc.values()) / len(pc)
            mr = sum(v["recall"] for v in pc.values()) / len(pc)
            L += [f"## {m.name} — {name}", "",
                  f"Accuracy {rep['accuracy']:.3f} · Macro-P {mp:.3f} · Macro-R {mr:.3f} · Macro-F1 {rep['macro_f1']:.3f}",
                  "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
            L += [f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
                  for c, v in pc.items()]
            if rep["confusion"]:
                L += ["", "Confusions (gold->pred: n): " + ", ".join(f"{k}: {v}" for k, v in rep["confusion"].items())]
            L.append("")
    out = ROOT / "reports" / "midproject_per_class.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
