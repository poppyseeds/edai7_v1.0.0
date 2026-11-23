"""Optional Gemini explanations. Never used for numerical decisions."""

from __future__ import annotations

import json
from typing import Any

from app.config import get_settings
from app.utils.logging_config import get_logger

logger = get_logger("LLMReasoner")


class LLMReasoner:
    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = bool(settings.gemini_api_key)
        self.model_name = settings.gemini_model
        self.api_key = settings.gemini_api_key

    def explain(self, payload: dict[str, Any], task: str) -> str:
        if not self.enabled:
            return self._fallback(payload, task)
        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            prompt = (
                "You are explaining an autonomous synthetic-data pipeline to a student. "
                "Use only the JSON facts provided. Do not invent metrics. "
                "Return 2-4 short sentences in plain language. When using a technical term, "
                "immediately explain it in simple words in parentheses.\n"
                f"Task: {task}\n"
                f"JSON:\n{json.dumps(payload, default=str)[:8000]}"
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None) or str(response)
            return text.strip()
        except Exception as exc:
            logger.warning("Gemini explanation failed: %s", exc)
            return self._fallback(payload, task)

    def explain_final_report(self, payload: dict[str, Any], fallback: str) -> str:
        """Explain a selected result only; deterministic selection has already happened."""

        if not self.enabled:
            return fallback
        try:
            from google import genai

            prompt = (
                "You are the explanation layer of a synthetic-data platform. All values in the "
                "JSON were computed deterministically. Do not calculate, change, invent, or round "
                "metrics. Do not claim anonymity, formal privacy, compliance, or differential privacy. "
                "Explain the selected result, rejected candidates, privacy limitations, and next actions "
                "in short titled sections for a technical reader. If a fact is absent, say it is unavailable.\n"
                f"JSON:\n{json.dumps(payload, default=str)[:12000]}"
            )
            response = genai.Client(api_key=self.api_key).models.generate_content(
                model=self.model_name, contents=prompt
            )
            text = (getattr(response, "text", None) or "").strip()
            # A missing or malformed response must never replace the deterministic report.
            return text if len(text) >= 80 else fallback
        except Exception as exc:
            logger.warning("Gemini final report failed: %s", exc)
            return fallback

    def _fallback(self, payload: dict[str, Any], task: str) -> str:
        if task == "analysis":
            issues = payload.get("issues") or []
            return (
                "Deterministic analysis found: "
                + (", ".join(issues) if issues else "no major structural issues")
                + ". Generation will target these problems without changing the evaluation split."
            )
        if task == "plan":
            return (
                f"Selected {payload.get('generator')} because {payload.get('reason', 'the planner rules matched the data profile')}."
            )
        if task == "validation":
            return (
                f"Overall validation score is {payload.get('overall_score')}. "
                "Scores are computed from KS/TV fidelity, correlation distance, diversity, and basic privacy indicators."
            )
        if task == "iteration":
            improved = payload.get("improved")
            return (
                "This iteration improved the held-out downstream metric."
                if improved
                else "This iteration did not beat the baseline; the optimizer may try another generator."
            )
        return "LLM explanation unavailable; showing deterministic pipeline results only."
