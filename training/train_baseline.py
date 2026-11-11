"""Train the TF-IDF + Logistic Regression classifier on all labeled data and save it
to models/classifier_tfidf.pkl (used by the web app and CLI).

    python training/train_baseline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_assistant.classification import TfidfClassifier, doc_text  # noqa: E402
from campus_assistant.pipeline import read_jsonl  # noqa: E402

if __name__ == "__main__":
    recs = read_jsonl(ROOT / "data" / "announcements.jsonl")
    clf = TfidfClassifier("logreg").fit([doc_text(r) for r in recs], [r["category"] for r in recs])
    out = ROOT / "models" / "classifier_tfidf.pkl"
    clf.save(out)
    print(f"Saved {clf.name} trained on {len(recs)} announcements -> {out}")
