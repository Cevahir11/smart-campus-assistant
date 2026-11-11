"""Mid-project experiment (CSE 484/654): frozen train/dev/test split, baselines with 95% bootstrap
intervals, baseline error dump, and (optionally) a first BERTurk number on the dev set.

    python evaluation/midproject.py                    # baselines -> reports/midproject_results.json/.md
    python evaluation/midproject.py --with-berturk     # + BERTurk fine-tuning, 3 seeds, dev set only

Protocol (fixed before any method result):
  * split   : stratified by category, 60/20/20, seed 2026 -> data/splits/midproject_split.json (frozen)
  * test A  : synthetic held-out test (84);  test B : hand-written challenge set (24, never trained on)
  * metric  : Macro-F1 (primary), accuracy;  95% CI = 1000 bootstrap resamples of the test set (seed 0)
  * "real"  : a method beats the baseline only if dMacro-F1 >= 0.03 AND the paired-bootstrap 95% CI
              of the difference excludes 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_assistant.classification import CATEGORIES, KeywordClassifier, TfidfClassifier, doc_text  # noqa
from campus_assistant.extraction import RuleExtractor  # noqa: E402
from campus_assistant.metrics import classification_report  # noqa: E402
from campus_assistant.pipeline import read_jsonl  # noqa: E402

SPLIT_SEED, BOOT_SEED, N_BOOT = 2026, 0, 1000
SPLIT_FILE = ROOT / "data" / "splits" / "midproject_split.json"


def make_or_load_split(records):
    if SPLIT_FILE.exists():
        ids = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    else:
        rng = random.Random(SPLIT_SEED)
        by = defaultdict(list)
        for r in records:
            by[r["category"]].append(r["id"])
        ids = {"train": [], "dev": [], "test": []}
        for c in CATEGORIES:
            xs = sorted(by[c])
            rng.shuffle(xs)
            k = round(len(xs) * 0.2)
            ids["test"] += xs[:k]
            ids["dev"] += xs[k:2 * k]
            ids["train"] += xs[2 * k:]
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(ids, indent=1), encoding="utf-8")
    by_id = {r["id"]: r for r in records}
    return {k: [by_id[i] for i in v] for k, v in ids.items()}


def macro_f1(gold, pred):
    return classification_report(gold, pred, CATEGORIES)["macro_f1"]


def bootstrap(gold, pred, seed=BOOT_SEED, n=N_BOOT):
    rng = random.Random(seed)
    idx = range(len(gold))
    f1s, accs = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        g, p = [gold[i] for i in s], [pred[i] for i in s]
        f1s.append(macro_f1(g, p))
        accs.append(sum(a == b for a, b in zip(g, p)) / len(g))
    f1s.sort(), accs.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    return {"f1_ci": [round(f1s[lo], 3), round(f1s[hi], 3)], "acc_ci": [round(accs[lo], 3), round(accs[hi], 3)]}


class Majority:
    name = "majority"

    def fit(self, texts, labels):
        from collections import Counter
        self.c = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.c] * len(texts)


def evaluate(model, rows):
    gold = [r["category"] for r in rows]
    pred = model.predict([doc_text(r) for r in rows])
    rep = classification_report(gold, pred, CATEGORIES)
    out = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"], "n": len(rows)}
    out.update(bootstrap(gold, pred))
    return out, pred


def run_berturk(train, dev, seeds, model_name, epochs, out_dir):
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)
    l2i = {c: i for i, c in enumerate(CATEGORIES)}
    tok = AutoTokenizer.from_pretrained(model_name)

    def ds(rows):
        d = Dataset.from_dict({"text": [doc_text(r) for r in rows], "label": [l2i[r["category"]] for r in rows]})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    dtr, ddev = ds(train), ds(dev)
    res = []
    for seed in seeds:
        set_seed(seed)
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(CATEGORIES), id2label=dict(enumerate(CATEGORIES)), label2id=l2i)
        args = TrainingArguments(output_dir=str(out_dir / f"_ckpt_{seed}"), num_train_epochs=epochs,
                                 learning_rate=3e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32,
                                 eval_strategy="no", save_strategy="no", weight_decay=0.01,
                                 warmup_steps=max(1, int(0.1 * epochs * -(-len(train) // 16))),
                                 logging_steps=50, report_to=[], seed=seed, use_cpu=not torch.cuda.is_available())
        tr = Trainer(model=model, args=args, train_dataset=dtr, data_collator=DataCollatorWithPadding(tok))
        tr.train()
        pred_ids = np.argmax(tr.predict(ddev).predictions, axis=-1)
        pred = [CATEGORIES[i] for i in pred_ids]
        gold = [r["category"] for r in dev]
        rep = classification_report(gold, pred, CATEGORIES)
        res.append({"seed": seed, "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
                    "minutes": round((time.time() - t0) / 60, 1),
                    "errors": [{"id": r["id"], "gold": g, "pred": p} for r, g, p in zip(dev, gold, pred) if g != p]})
        print(f"  BERTurk seed {seed}: dev acc {rep['accuracy']:.3f}  macro-F1 {rep['macro_f1']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-berturk", action="store_true")
    ap.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--seeds", default="13,42,2026")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    records = read_jsonl(ROOT / "data" / "announcements.jsonl")
    challenge = read_jsonl(ROOT / "data" / "challenge_set.jsonl")
    sp = make_or_load_split(records)
    train, dev, test = sp["train"], sp["dev"], sp["test"]
    print(f"split: train {len(train)} / dev {len(dev)} / test {len(test)} / challenge {len(challenge)}")

    res = {"split": {k: len(v) for k, v in sp.items()} | {"challenge": len(challenge)},
           "protocol": {"split_seed": SPLIT_SEED, "bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED,
                        "min_real_diff_macro_f1": 0.03},
           "baselines": {}, "errors": []}
    models = [Majority(), KeywordClassifier(), TfidfClassifier("nb"), TfidfClassifier("svm"),
              TfidfClassifier("logreg")]
    for m in models:
        m.fit([doc_text(r) for r in train], [r["category"] for r in train])
        row = {}
        for name, rows in [("dev", dev), ("test", test), ("challenge", challenge)]:
            row[name], pred = evaluate(m, rows)
            if m.name in ("keyword_rules", "tfidf_logreg") and name != "test":
                res["errors"] += [{"model": m.name, "set": name, "id": r["id"], "lang": r["lang"],
                                   "gold": r["category"], "pred": p, "title": r["title"], "text": r["text"]}
                                  for r, p in zip(rows, pred) if p != r["category"]]
        res["baselines"][m.name] = row
        print(f"{m.name:14s} dev {row['dev']['macro_f1']:.3f}  test {row['test']['macro_f1']:.3f} "
              f"{row['test']['f1_ci']}  challenge {row['challenge']['macro_f1']:.3f} {row['challenge']['f1_ci']}")

    # keyword rules need no training -> its errors on the training split are also fair to analyse
    kw = KeywordClassifier().fit([], ["exam"])
    for r, p in zip(train, kw.predict([doc_text(r) for r in train])):
        if p != r["category"]:
            res["errors"].append({"model": "keyword_rules", "set": "train", "id": r["id"], "lang": r["lang"],
                                  "gold": r["category"], "pred": p, "title": r["title"], "text": r["text"]})

    # extraction baseline errors on the hand-written set (stage 02, for the pipeline view)
    ex = RuleExtractor()
    for r in challenge:
        gold = {(e["label"], e["norm"].lower()) for e in r["entities"]}
        pred = {(e.label, e.norm.lower()) for e in ex.extract(r["text"], date.fromisoformat(r["published"]))}
        for lab, v in sorted(gold - pred):
            res["errors"].append({"model": "rule_extractor", "set": "challenge", "id": r["id"], "lang": r["lang"],
                                  "gold": f"{lab}={v}", "pred": "missed", "title": r["title"], "text": r["text"]})
        for lab, v in sorted(pred - gold):
            res["errors"].append({"model": "rule_extractor", "set": "challenge", "id": r["id"], "lang": r["lang"],
                                  "gold": "none", "pred": f"{lab}={v}", "title": r["title"], "text": r["text"]})

    if args.with_berturk:
        seeds = [int(s) for s in args.seeds.split(",")]
        res["berturk_dev"] = run_berturk(train, dev, seeds, args.model, args.epochs, ROOT / "models")

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    prev = out / "midproject_results.json"
    if not args.with_berturk and prev.exists():  # keep the (slow) BERTurk rows from an earlier run
        old_res = json.loads(prev.read_text(encoding="utf-8"))
        if "berturk_dev" in old_res:
            res["berturk_dev"] = old_res["berturk_dev"]
    (out / "midproject_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"errors dumped: {len(res['errors'])}  -> reports/midproject_results.json")


if __name__ == "__main__":
    main()
