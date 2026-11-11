"""Optional Claude integration (prompted LLM for summarization and grounded QA).

Enabled only when the `anthropic` package is installed and credentials are available
(ANTHROPIC_API_KEY). Every caller has a rule-based fallback, so the project runs fully
offline without it.
"""
from __future__ import annotations

import os

DEFAULT_MODEL = os.environ.get("CAMPUS_LLM_MODEL", "claude-opus-5-5")


class ClaudeLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        import anthropic  # optional dependency
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    @staticmethod
    def available() -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str | None:
        """Return the text answer, or None on refusal / API error (caller falls back)."""
        try:
            kwargs = dict(model=self.model, max_tokens=max_tokens,
                          output_config={"effort": "low"},
                          messages=[{"role": "user", "content": prompt}])
            if system:
                kwargs["system"] = system
            if self.model in ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"):
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except self.anthropic.APIConnectionError as exc:
            print(f"[llm] connection error: {exc}")
            return None
        except self.anthropic.RateLimitError as exc:
            print(f"[llm] rate limited: {exc}")
            return None
        except self.anthropic.APIStatusError as exc:
            print(f"[llm] API error {exc.status_code}: {exc.message}")
            return None
        if resp.stop_reason == "refusal":
            return None
        return "".join(b.text for b in resp.content if b.type == "text").strip() or None
