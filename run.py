"""Start the web assistant:  python run.py   ->  http://127.0.0.1:5000"""
import argparse

from campus_assistant.web import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)
