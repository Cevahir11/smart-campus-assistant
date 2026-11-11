"""Fine-tune a token-classification (NER) model for DATE / TIME / LOCATION / COURSE
(proposal step 02, "NER + rules"). Character-level entity spans from
data/announcements.jsonl are converted to BIO tags over sub-word tokens.

    pip install -r requirements-advanced.txt
    python training/train_ner.py --model dbmdz/bert-base-turkish-cased

The model is saved to models/ner; campus_assistant.extraction.load_extractor() then
returns the HybridExtractor (rules + model) automatically.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_assistant.pipeline import read_jsonl  # noqa: E402
from evaluation.evaluate import stratified_split  # noqa: E402

LABELS = ["O", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-LOCATION", "I-LOCATION", "B-COURSE", "I-COURSE"]


def main():
    import numpy as np
    from datasets import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer, TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default=str(ROOT / "models" / "ner"))
    args = ap.parse_args()

    l2i = {l: i for i, l in enumerate(LABELS)}
    tok = AutoTokenizer.from_pretrained(args.model)
    recs = read_jsonl(ROOT / "data" / "announcements.jsonl")
    train, test = stratified_split(recs)

    def encode(rows):
        enc = tok([r["text"] for r in rows], truncation=True, max_length=256, return_offsets_mapping=True)
        all_labels = []
        for r, offsets in zip(rows, enc["offset_mapping"]):
            labels = []
            for s, e in offsets:
                if s == e:
                    labels.append(-100)
                    continue
                tag = "O"
                for ent in r["entities"]:
                    if s >= ent["start"] and e <= ent["end"]:
                        tag = ("B-" if s == ent["start"] else "I-") + ent["label"]
                        break
                labels.append(l2i[tag])
            all_labels.append(labels)
        enc.pop("offset_mapping")
        enc["labels"] = all_labels
        return Dataset.from_dict(dict(enc))

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=l2i)

    def metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        mask = p.label_ids != -100
        gold, pr = p.label_ids[mask], pred[mask]
        ent = gold != 0
        tp = int(((pr == gold) & ent).sum())
        prec = tp / max(1, int((pr != 0).sum()))
        rec = tp / max(1, int(ent.sum()))
        return {"token_precision": prec, "token_recall": rec,
                "token_f1": 2 * prec * rec / max(1e-9, prec + rec)}

    targs = TrainingArguments(output_dir=str(ROOT / "models" / "_ckpt_ner"), num_train_epochs=args.epochs,
                              learning_rate=args.lr, per_device_train_batch_size=16, eval_strategy="epoch",
                              save_strategy="no", weight_decay=0.01, logging_steps=10, report_to=[])
    trainer = Trainer(model=model, args=targs, train_dataset=encode(train), eval_dataset=encode(test),
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=metrics)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
