"""Metrics implemented without extra dependencies: P/R/F1, accuracy, macro-F1, ROUGE-1/2/L."""
from __future__ import annotations

from collections import Counter

from .textutils import tokenize


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def classification_report(gold, pred, labels):
    per = {}
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        m = prf(tp, fp, fn)
        per[c] = {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                  "support": sum(1 for g in gold if g == c)}
    present = [c for c in labels if per[c]["support"] > 0]
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0
    macro = sum(per[c]["f1"] for c in present) / len(present) if present else 0.0
    return {"accuracy": round(acc, 4), "macro_f1": round(macro, 4), "per_class": per,
            "confusion": dict(Counter(f"{g}->{p}" for g, p in zip(gold, pred) if g != p))}


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _f(overlap, a, b):
    if not a or not b or not overlap:
        return 0.0
    p, r = overlap / a, overlap / b
    return 2 * p * r / (p + r)


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(pred: str, ref: str) -> dict:
    p, r = tokenize(pred), tokenize(ref)
    out = {}
    for n in (1, 2):
        pn, rn = _ngrams(p, n), _ngrams(r, n)
        out[f"rouge{n}"] = _f(sum((pn & rn).values()), sum(pn.values()), sum(rn.values()))
    out["rougeL"] = _f(_lcs(p, r), len(p), len(r))
    return out
