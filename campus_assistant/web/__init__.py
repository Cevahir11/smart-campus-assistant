"""Flask web app: chat assistant + structured announcement feed."""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from ..service import AssistantService


def create_app(service: AssistantService | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_AS_ASCII"] = False
    svc = service or AssistantService()
    app.config["SERVICE"] = svc

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/stats")
    def stats():
        return jsonify(svc.stats())

    @app.get("/api/announcements")
    def announcements():
        return jsonify(svc.list(category=request.args.get("category") or None, q=request.args.get("q") or None,
                                upcoming=request.args.get("upcoming") == "1",
                                limit=int(request.args.get("limit", 200))))

    @app.post("/api/ask")
    def ask():
        body = request.get_json(force=True, silent=True) or {}
        question = (body.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        return jsonify(svc.ask(question, mode=body.get("mode", "rules")))

    @app.post("/api/analyze")
    def analyze():
        body = request.get_json(force=True, silent=True) or {}
        if not (body.get("text") or "").strip():
            return jsonify({"error": "text is required"}), 400
        return jsonify(svc.analyze(body.get("title", ""), body["text"], body.get("published")))

    @app.post("/api/announcements")
    def add():
        body = request.get_json(force=True, silent=True) or {}
        if not (body.get("text") or "").strip():
            return jsonify({"error": "text is required"}), 400
        return jsonify(svc.add(body.get("title", ""), body["text"], body.get("source", "user"),
                               body.get("published"))), 201

    return app
