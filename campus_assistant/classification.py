"""Announcement category classification.

Models (all expose fit(texts, labels) / predict(texts) / predict_proba-like scores):

* KeywordClassifier     - hand-written keyword rules (simplest baseline)
* TfidfClassifier       - TF-IDF (word + char n-grams) + Logistic Regression / Linear SVM / Naive Bayes
* TransformerClassifier - fine-tuned BERTurk / XLM-R (see training/train_transformer_classifier.py)
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

from .textutils import fold

CATEGORIES = ["exam", "event", "scholarship", "cancellation", "room_change", "registration"]
CATEGORY_TR = {"exam": "Sınav", "event": "Etkinlik", "scholarship": "Burs",
               "cancellation": "Ders İptali", "room_change": "Derslik Değişikliği",
               "registration": "Kayıt / Akademik Takvim"}
CATEGORY_EN = {"exam": "Exam", "event": "Event", "scholarship": "Scholarship",
               "cancellation": "Class cancellation", "room_change": "Room change",
               "registration": "Registration / Calendar"}

# keywords are matched on folded text (lowercase, no diacritics)
KEYWORDS = {
    "exam": ["sinav", "vize", "final", "quiz", "butunleme", "mazeret sinavi", "exam", "midterm", "sinav suresi"],
    "event": ["etkinlik", "seminer", "soylesi", "konser", "atolye", "workshop", "kulup", "davet", "panel",
              "turnuva", "sergi", "event", "festival", "hackathon", "join us", "invites"],
    "scholarship": ["burs", "basvuru", "erasmus", "tubitak", "destek", "scholarship", "stipend", "deadline",
                    "hareketlilik", "degisim programi"],
    "cancellation": ["iptal", "yapilmayacak", "yapilamayacak", "telafi", "cancel", "no lecture",
                     "make-up class"],
    "room_change": ["derslik degisikligi", "yerine", "tasinmistir", "salon degis", "guncellenmistir",
                    "room change", "instead of", "moved", "derslikte yapilacak"],
    "registration": ["kayit", "ekleme-birakma", "akademik takvim", "harc", "ogrenci bilgi sistemi",
                     "danisman onayi", "registration", "add-drop", "tuition", "yatay gecis", "muafiyet",
                     "cift anadal", "yandal", "mezuniyet"],
}


class KeywordClassifier:
    name = "keyword_rules"

    def fit(self, texts, labels):
        self.prior = Counter(labels).most_common(1)[0][0]
        return self

    def scores(self, text):
        t = fold(text)
        return {c: sum(t.count(k) for k in kws) for c, kws in KEYWORDS.items()}

    def predict(self, texts):
        out = []
        for t in texts:
            s = self.scores(t)
            best = max(s, key=s.get)
            out.append(best if s[best] > 0 else getattr(self, "prior", "exam"))
        return out

    def predict_proba(self, texts):
        res = []
        for t in texts:
            s = self.scores(t)
            tot = sum(s.values()) or 1
            res.append({c: v / tot for c, v in s.items()})
        return res


class TfidfClassifier:
    """TF-IDF over words + character n-grams (robust to Turkish suffixes)."""

    def __init__(self, algo: str = "logreg"):
        self.algo = algo
        self.name = f"tfidf_{algo}"

    def _build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import ComplementNB
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC

        feats = FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=fold, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(preprocessor=fold, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                     sublinear_tf=True)),
        ])
        clf = {"logreg": LogisticRegression(max_iter=2000, C=5.0),
               "svm": LinearSVC(C=1.0),
               "nb": ComplementNB(alpha=0.3)}[self.algo]
        return Pipeline([("feats", feats), ("clf", clf)])

    def fit(self, texts, labels):
        self.pipe = self._build()
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict(self, texts):
        return list(self.pipe.predict(list(texts)))

    def predict_proba(self, texts):
        clf = self.pipe.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            probs = self.pipe.predict_proba(list(texts))
            return [dict(zip(clf.classes_, map(float, p))) for p in probs]
        import numpy as np
        dec = self.pipe.decision_function(list(texts))
        out = []
        for row in dec:
            e = np.exp(row - row.max())
            out.append(dict(zip(clf.classes_, map(float, e / e.sum()))))
        return out

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


class TransformerClassifier:
    """Inference wrapper for a fine-tuned BERTurk / XLM-R sequence classifier."""
    name = "transformer"

    def __init__(self, model_dir):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=str(model_dir), truncation=True, top_k=None)

    def fit(self, texts, labels):  # trained offline
        return self

    def predict(self, texts):
        return [max(p, key=lambda x: x["score"])["label"] for p in self.pipe(list(texts))]

    def predict_proba(self, texts):
        return [{x["label"]: float(x["score"]) for x in p} for p in self.pipe(list(texts))]


def doc_text(rec: dict) -> str:
    return f"{rec.get('title', '')}. {rec.get('text', '')}"


def load_classifier(models_dir=None, train_records=None):
    """Best available classifier: fine-tuned transformer > saved TF-IDF > TF-IDF trained now."""
    models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parents[1] / "models"
    tdir = models_dir / "classifier_transformer"
    if (tdir / "config.json").exists():
        try:
            return TransformerClassifier(tdir)
        except Exception as exc:
            print(f"[classification] transformer not loaded ({exc}); falling back to TF-IDF")
    pkl = models_dir / "classifier_tfidf.pkl"
    if pkl.exists():
        try:
            return TfidfClassifier.load(pkl)
        except Exception as exc:
            print(f"[classification] could not load {pkl} ({exc}); retraining")
    if train_records:
        return TfidfClassifier("logreg").fit([doc_text(r) for r in train_records],
                                             [r["category"] for r in train_records])
    return KeywordClassifier().fit([], ["exam"])
