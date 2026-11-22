"""Optional Claude layer.

VolSentinel runs end to end without a language model: tool choice, scoring and
guardrails are deterministic. When the `anthropic` SDK is installed and
credentials are available (ANTHROPIC_API_KEY, or VOLSENTINEL_LLM=on with an
`ant auth login` profile), Claude drafts the Scenario Analyst's explanation and
gives the Risk Critic's 30% judgement score (specificity, clarity). Every LLM
output still passes the deterministic citation and language checks, and any LLM
failure falls back to the deterministic path.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)


class LLM:
    def __init__(self, settings):
        self.model = settings.llm_model
        self.client = None
        self.reason = ""
        mode = os.environ.get("VOLSENTINEL_LLM", "auto").lower()
        if mode == "off" or not settings.llm_enabled:
            self.reason = "disabled (VOLSENTINEL_LLM=off)"
            return
        try:
            import anthropic  # noqa: F401
        except ImportError:
            self.reason = "anthropic SDK not installed (pip install anthropic)"
            return
        if mode != "on" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            self.reason = "no credentials (set ANTHROPIC_API_KEY, or VOLSENTINEL_LLM=on to use an ant profile)"
            return
        import anthropic
        self.client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        self.reason = f"enabled ({self.model})"

    @property
    def available(self) -> bool:
        return self.client is not None

    def describe(self) -> str:
        return self.reason

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 4000) -> dict | None:
        """One structured call. Returns the parsed object, or None on any failure or refusal."""
        if not self.client:
            return None
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if resp.stop_reason == "refusal":
                log.warning("LLM refused: %s", getattr(resp, "stop_details", None))
                return None
            text = next(b.text for b in resp.content if b.type == "text")
            return json.loads(text)
        except Exception as e:  # noqa: BLE001 - deterministic path takes over
            log.warning("LLM call failed: %s: %s", type(e).__name__, e)
            return None
