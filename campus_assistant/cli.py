"""Command line interface.

    python -m campus_assistant.cli ask "Bu hafta sınav var mı?"
    python -m campus_assistant.cli analyze --title "CENG 301 sınavı" --text "Sınav 20 Kasım 10:00'da B-204'te."
    python -m campus_assistant.cli chat          # interactive
"""
import argparse
import json

from .service import AssistantService


def main():
    ap = argparse.ArgumentParser(prog="campus_assistant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--llm", action="store_true")
    n = sub.add_parser("analyze")
    n.add_argument("--title", default="")
    n.add_argument("--text", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    svc = AssistantService()
    if args.cmd == "ask":
        res = svc.ask(args.question, mode="llm" if args.llm else "rules")
        print(res["answer"])
        for s in res["sources"][:5]:
            print(f"  - [{s['id']}] {s['title']}")
    elif args.cmd == "analyze":
        r = svc.analyze(args.title, args.text)
        print(json.dumps({k: r[k] for k in ("category", "confidence", "fields", "summary")}, ensure_ascii=False,
                         indent=2))
    else:
        print(f"Referans tarih: {svc.today} — çıkmak için boş satır")
        while True:
            q = input("> ").strip()
            if not q:
                break
            print(svc.ask(q, mode="llm" if args.llm else "rules")["answer"])


if __name__ == "__main__":
    main()
