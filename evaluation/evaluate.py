"""Run the full comparative evaluation from the proposal and write a report.

    Classification   : Accuracy, Macro-F1      (keyword rules vs TF-IDF NB / SVM / LogReg [vs transformer])
    Extraction       : Precision / Recall / F1 per entity type   (rules [vs hybrid NER])
    Summarization    : ROUGE-1/2/L F1           (lead-1 vs centroid vs structured template [vs mT5 / LLM])
    Question answer. : Answer accuracy on the QA set (~50 questions), retrieval hit@1

Usage:
    python evaluation/evaluate.py                 # offline baselines
    python evaluation/evaluate.py --with-transformers --with-llm
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_assistant.classification import (CATEGORIES, KeywordClassifier,  # noqa: E402
                                             TfidfClassifier, TransformerClassifier, doc_text)
from campus_assistant.extraction import RuleExtractor, load_extractor  # noqa: E402
from campus_assistant.metrics import classification_report, prf, rouge  # noqa: E402
from campus_assistant.pipeline import (AnnouncementProcessor, dataset_reference_date,  # noqa: E402
                                       read_jsonl)
from campus_assistant.qa import QAEngine  # noqa: E402
from campus_assistant.summarization import (CentroidSummarizer, LeadSummarizer,  # noqa: E402
                                            StructuredSummarizer)
from campus_assistant.textutils import format_date_short  # noqa: E402


def stratified_split(records, test_frac=0.2, seed=13):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in records:
        by[r["category"]].append(r)
    train, test = [], []
    for rs in by.values():
        rs = rs[:]
        rng.shuffle(rs)
        k = max(1, round(len(rs) * test_frac))
        test += rs[:k]
        train += rs[k:]
    return train, test


def kfold_predict(records, make_model, k=5, seed=13):
    """Out-of-fold category predictions for every record."""
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    preds = [None] * len(records)
    for fold_i in range(k):
        test_idx = set(idx[fold_i::k])
        tr = [records[i] for i in idx if i not in test_idx]
        model = make_model().fit([doc_text(r) for r in tr], [r["category"] for r in tr])
        te = sorted(test_idx)
        for i, p in zip(te, model.predict([doc_text(records[i]) for i in te])):
            preds[i] = p
    return preds


# ----------------------------------------------------------------------------- tasks
def eval_classification(train, test, extra_models):
    models = [KeywordClassifier(), TfidfClassifier("nb"), TfidfClassifier("svm"), TfidfClassifier("logreg")]
    models += extra_models
    rows = {}
    for m in models:
        t0 = time.time()
        m.fit([doc_text(r) for r in train], [r["category"] for r in train])
        pred = m.predict([doc_text(r) for r in test])
        rep = classification_report([r["category"] for r in test], pred, CATEGORIES)
        rep["seconds"] = round(time.time() - t0, 2)
        rows[m.name] = rep
    return rows


def eval_extraction(records, extractors):
    out = {}
    for ex in extractors:
        tp, fp, fn = Counter(), Counter(), Counter()
        for r in records:
            gold = {(e["label"], e["norm"].lower()) for e in r["entities"]}
            pred = {(e.label, e.norm.lower()) for e in ex.extract(r["text"], date.fromisoformat(r["published"]))}
            for lab, _ in pred & gold:
                tp[lab] += 1
            for lab, _ in pred - gold:
                fp[lab] += 1
            for lab, _ in gold - pred:
                fn[lab] += 1
        res = {lab: prf(tp[lab], fp[lab], fn[lab]) for lab in sorted(set(tp) | set(fp) | set(fn))}
        res["micro"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
        out[ex.name] = res
    return out


def eval_summarization(processed, gold_by_id, summarizers):
    out = {}
    for s in summarizers:
        scores = defaultdict(list)
        for r in processed:
            pred = s.summarize(r)
            for k, v in rouge(pred, gold_by_id[r["id"]]["summary"]).items():
                scores[k].append(v)
        out[s.name] = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
    return out


def fact_ok(fact, answer, lang, found):
    t, v = fact["type"], fact["value"]
    low = answer.lower()
    if t == "NO_ANSWER":
        return not found
    if t == "DATE":
        d = date.fromisoformat(v)
        return format_date_short(d, lang).lower() in low or format_date_short(d, "tr").lower() in low \
            or format_date_short(d, "en").lower() in low
    if t == "TEXT_ANY":
        return any(x.lower() in low for x in v)
    return str(v).lower() in low


def eval_qa(engine, questions, mode="rules"):
    per_type = defaultdict(lambda: [0, 0])
    correct = hit1 = 0
    details = []
    for q in questions:
        res = engine.answer(q["question"], mode=mode)
        ok = all(fact_ok(f, res["answer"], q["lang"], res["found"]) for f in q["expected_facts"])
        src = [s["id"] for s in res["sources"]]
        h = (not q["expected_ids"] and not src) or (bool(src) and src[0] in q["expected_ids"])
        correct += ok
        hit1 += h
        per_type[q["type"]][0] += ok
        per_type[q["type"]][1] += 1
        details.append({"id": q["id"], "question": q["question"], "correct": ok, "answer": res["answer"],
                        "expected": q["expected_facts"], "top_source": src[:1]})
    n = len(questions)
    return {"accuracy": round(correct / n, 4), "retrieval_hit@1": round(hit1 / n, 4), "n": n,
            "by_type": {k: f"{a}/{b}" for k, (a, b) in sorted(per_type.items())}, "details": details}


# ----------------------------------------------------------------------------- report
def md_report(res, meta):
    L = ["# Smart Campus Announcement Assistant — Evaluation Report", ""]
    L.append(f"Dataset: {meta['n_announcements']} announcements, {meta['n_questions']} QA questions, "
             f"reference date {meta['base_date']}" + (" (synthetic seed data)" if meta.get("synthetic") else ""))
    L += ["", "## 1. Classification (held-out 20%)", "", "| Model | Accuracy | Macro-F1 |", "|---|---|---|"]
    for name, r in res["classification"].items():
        L.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    best = max(res["classification"], key=lambda k: res["classification"][k]["macro_f1"])
    L += ["", f"Per-class F1 ({best}):", "", "| Class | P | R | F1 | Support |", "|---|---|---|---|---|"]
    for c, v in res["classification"][best]["per_class"].items():
        L.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    L += ["", "## 2. Information extraction", ""]
    for name, r in res["extraction"].items():
        L += [f"**{name}**", "", "| Entity | Precision | Recall | F1 |", "|---|---|---|---|"]
        for lab, v in r.items():
            L.append(f"| {lab} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
        L.append("")
    if "extraction_challenge" in res:
        L += ["**Hand-written challenge set (real-world style, not template-generated)**", ""]
        for name, r in res["extraction_challenge"].items():
            L += [f"_{name}_", "", "| Entity | Precision | Recall | F1 |", "|---|---|---|---|"]
            for lab, v in r.items():
                L.append(f"| {lab} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
            L.append("")
    if "classification_challenge" in res:
        L += ["**Challenge-set classification**", "", "| Model | Accuracy | Macro-F1 |", "|---|---|---|"]
        for name, r in res["classification_challenge"].items():
            L.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
        L.append("")
    L += ["## 3. Summarization (ROUGE F1 vs reference one-sentence summaries)", "",
          "| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |", "|---|---|---|---|"]
    for name, r in res["summarization"].items():
        L.append(f"| {name} | {r['rouge1']:.3f} | {r['rouge2']:.3f} | {r['rougeL']:.3f} |")
    L += ["", "## 4. Question answering", "", "| Mode | Answer accuracy | Retrieval hit@1 | N |", "|---|---|---|---|"]
    for mode, r in res["qa"].items():
        L.append(f"| {mode} | {r['accuracy']:.3f} | {r['retrieval_hit@1']:.3f} | {r['n']} |")
    for mode, r in res["qa"].items():
        L += ["", f"By question type ({mode}): " + ", ".join(f"{k} {v}" for k, v in r["by_type"].items())]
        wrong = [d for d in r["details"] if not d["correct"]]
        if wrong:
            L += ["", "Errors:", ""]
            for d in wrong[:15]:
                L.append(f"- *{d['question']}* → {d['answer'].splitlines()[0][:160]}  (expected {d['expected']})")
    L += ["", "## Notes", "",
          "- Scores on the synthetic seed data are optimistic: the rules and templates share a vocabulary. "
          "The challenge set and, above all, the planned 300–500 manually annotated real announcements are the "
          "numbers to report.",
          "- QA runs end to end on *predicted* categories (5-fold out-of-fold) and *extracted* fields, "
          "not on gold labels."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--with-transformers", action="store_true",
                    help="also evaluate fine-tuned models in models/ (classifier_transformer, ner) and mT5")
    ap.add_argument("--with-llm", action="store_true", help="also evaluate Claude (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    data = Path(args.data)
    records = read_jsonl(data / "announcements.jsonl")
    questions = read_jsonl(data / "qa_eval.jsonl")
    meta = json.loads((data / "meta.json").read_text())
    today = dataset_reference_date(data)
    train, test = stratified_split(records)
    res = {}

    extra_cls, extractors = [], [RuleExtractor()]
    if args.with_transformers:
        tdir = ROOT / "models" / "classifier_transformer"
        if (tdir / "config.json").exists():
            extra_cls.append(TransformerClassifier(tdir))
        ex = load_extractor()
        if ex.name != "rules":
            extractors.append(ex)

    print("[1/4] classification ...")
    res["classification"] = eval_classification(train, test, extra_cls)
    print("[2/4] extraction ...")
    res["extraction"] = eval_extraction(records, extractors)
    challenge = data / "challenge_set.jsonl"
    if challenge.exists():
        ch = read_jsonl(challenge)
        res["extraction_challenge"] = eval_extraction(ch, extractors)
        cls = {}
        for m in [KeywordClassifier(), TfidfClassifier("logreg")] + extra_cls:
            m.fit([doc_text(r) for r in records], [r["category"] for r in records])
            cls[m.name] = classification_report([r["category"] for r in ch], m.predict([doc_text(r) for r in ch]),
                                                CATEGORIES)
        res["classification_challenge"] = cls

    print("[3/4] processing all announcements with out-of-fold categories ...")
    oof = kfold_predict(records, lambda: TfidfClassifier("logreg"))

    class Fixed:  # feed the out-of-fold prediction into the pipeline
        def __init__(self):
            self.next = None

        def predict_proba(self, texts):
            return [{self.next: 1.0}]

    fixed = Fixed()
    proc = AnnouncementProcessor(classifier=fixed, extractor=RuleExtractor())
    processed = []
    for r, p in zip(records, oof):
        fixed.next = p
        processed.append(proc.process(r))
    gold_by_id = {r["id"]: r for r in records}
    summarizers = [LeadSummarizer(), CentroidSummarizer(), StructuredSummarizer()]
    if args.with_transformers:
        try:
            from campus_assistant.summarization import Seq2SeqSummarizer
            summarizers.append(Seq2SeqSummarizer())
        except Exception as exc:
            print(f"   mT5 skipped: {exc}")
    llm = None
    if args.with_llm:
        from campus_assistant.llm import ClaudeLLM
        if ClaudeLLM.available():
            from campus_assistant.summarization import LLMSummarizer
            llm = ClaudeLLM()
            summarizers.append(LLMSummarizer(llm))
        else:
            print("   LLM skipped: set ANTHROPIC_API_KEY and pip install anthropic")
    res["summarization"] = eval_summarization(processed, gold_by_id, summarizers)

    print("[4/4] question answering ...")
    engine = QAEngine(processed, today=today, llm=llm)
    res["qa"] = {"rules": eval_qa(engine, questions, "rules")}
    if llm:
        res["qa"]["llm"] = eval_qa(engine, questions, "llm")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    report = md_report(res, meta)
    (out / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
