"""Fine-tune BERTurk or XLM-R for announcement classification (proposal step 01).

    pip install -r requirements-advanced.txt
    python training/train_transformer_classifier.py --model dbmdz/bert-base-turkish-cased
    python training/train_transformer_classifier.py --model xlm-roberta-base --epochs 5

The model is saved to models/classifier_transformer and is picked up automatically by
the app (campus_assistant.classification.load_classifier) and by
`python evaluation/evaluate.py --with-transformers`. A GPU is recommended but a few
epochs on ~400 short texts also run on CPU.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_assistant.classification import CATEGORIES, doc_text  # noqa: E402
from campus_assistant.pipeline import read_jsonl  # noqa: E402
from evaluation.evaluate import stratified_split  # noqa: E402


def main():
    import numpy as np
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "models" / "classifier_transformer"))
    args = ap.parse_args()

    recs = read_jsonl(ROOT / "data" / "announcements.jsonl")
    train, test = stratified_split(recs)
    l2i = {c: i for i, c in enumerate(CATEGORIES)}
    tok = AutoTokenizer.from_pretrained(args.model)

    def ds(rows):
        d = Dataset.from_dict({"text": [doc_text(r) for r in rows], "label": [l2i[r["category"]] for r in rows]})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(CATEGORIES), id2label=dict(enumerate(CATEGORIES)), label2id=l2i)

    def metrics(p):
        from sklearn.metrics import accuracy_score, f1_score
        pred = np.argmax(p.predictions, axis=-1)
        return {"accuracy": accuracy_score(p.label_ids, pred), "macro_f1": f1_score(p.label_ids, pred, average="macro")}

    targs = TrainingArguments(output_dir=str(ROOT / "models" / "_ckpt_cls"), num_train_epochs=args.epochs,
                              learning_rate=args.lr, per_device_train_batch_size=args.batch,
                              per_device_eval_batch_size=32, eval_strategy="epoch", save_strategy="no",
                              weight_decay=0.01, warmup_ratio=0.1, logging_steps=10, report_to=[])
    trainer = Trainer(model=model, args=targs, train_dataset=ds(train), eval_dataset=ds(test),
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
