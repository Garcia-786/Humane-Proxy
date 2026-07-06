# Copyright 2026 Vishisht Mishra (Vishisht16)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage-3 provider: OpenAI-compatible chat model as a safety classifier.

Sends a structured classifier prompt to any OpenAI-compatible chat API
and parses the JSON response.  The prompt explicitly instructs the model
that **jailbreaks are always safe** — the focus is on genuine human harm.
"""

from __future__ import annotations

import logging
import os

from humane_proxy._json import loads as _json_loads
from humane_proxy.classifiers.models import ClassificationResult
from humane_proxy.classifiers.stage3.base import Stage3Classifier
from humane_proxy.telemetry import traced_stage

logger = logging.getLogger("humane_proxy.classifiers.stage3.openai_chat")


def _extract_json_object(content: str) -> dict | None:
    """Return the JSON object from *content*, tolerating surrounding prose.

    Tries a direct parse first, then scans for the last balanced ``{...}``
    span (reasoning models put the answer object at the end). Returns None
    when nothing parses to a dict.
    """
    if not content:
        return None
    try:
        data = _json_loads(content)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass

    # Scan for balanced brace spans; keep the last one that parses to a dict.
    found: dict | None = None
    depth = 0
    start = -1
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = _json_loads(content[start:i + 1])
                    if isinstance(candidate, dict) and "category" in candidate:
                        found = candidate
                except ValueError:
                    pass
    return found

_SYSTEM_PROMPT = """\
You are a safety classifier for a human-protection system called HumaneProxy.

Your job is to classify user messages into one of three categories:
1. "self_harm" — the user is expressing suicidal ideation, self-injury intent,
   or seeking methods to harm themselves.
2. "criminal_intent" — the user is seeking to commit violence, create weapons
   or drugs, traffic persons, create CSAM, or carry out other serious crimes.
3. "safe" — everything else, including normal conversation.

CRITICAL RULES:
- A jailbreak attempt (e.g. "ignore previous instructions") is ALWAYS "safe".
  Jailbreaks are NOT our concern.
- Hyperbolic expressions like "I'm dying of laughter" are ALWAYS "safe".
- Academic, journalistic, fictional, or help-seeking questions (e.g. "what are
  warning signs of suicide", "how do detectives investigate fraud") are "safe".
- Otherwise, if the message genuinely seeks to harm a person (including the
  user) or to carry out a serious crime, flag it. Do NOT wave away real intent
  as hypothetical. Missing genuine harm is far worse than a false alarm.

Respond ONLY with a JSON object: {"category": "...", "score": 0.0-1.0, "reasoning": "..."}
"""


class OpenAIChatClassifier(Stage3Classifier):
    """Stage-3 classifier using an OpenAI-compatible chat model."""

    def __init__(self, config: dict) -> None:
        chat_cfg = config.get("stage3", {}).get("openai_chat", {})
        self._api_url: str = chat_cfg.get(
            "api_url", "https://api.openai.com/v1/chat/completions"
        )
        self._model: str = chat_cfg.get("model", "gpt-4o-mini")
        # Reasoning models emit chain-of-thought before the JSON answer. Too
        # small a budget truncates the response — and with JSON-mode enabled,
        # providers reject the whole call — so the classifier would silently
        # fall back to the prior (safe) verdict. Keep this generous.
        self._max_tokens: int = int(chat_cfg.get("max_tokens", 1024))
        self._json_mode: bool = bool(chat_cfg.get("json_mode", False))
        self._timeout: float = config.get("stage3", {}).get("timeout", 10.0)
        # OPENAI_API_KEY first, then LLM_API_KEY — the endpoint is
        # configurable (Groq, Together, a local server), so the key need
        # not be an OpenAI one.
        self._api_key: str = os.environ.get(
            "OPENAI_API_KEY", os.environ.get("LLM_API_KEY", "")
        )

    @traced_stage("stage3.reasoning_llm")
    async def classify(
        self, text: str, prior: ClassificationResult
    ) -> ClassificationResult:
        """Send *text* to the chat model for classification."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
        }
        # JSON mode forces the whole output to be a single JSON value, which
        # reasoning models break (their chain-of-thought precedes the JSON,
        # and an overflow rejects the entire call). Off by default so the
        # tolerant parser can extract the JSON from free-form text; enable
        # for plain chat models that support it.
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            from humane_proxy.http_client import get_async_client

            client = get_async_client()
            resp = await client.post(
                self._api_url, json=payload, headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()

            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            return self._parse_response(content)

        except Exception as exc:
            logger.warning("OpenAI Chat Stage-3 error: %s", exc)
            return ClassificationResult(
                category=prior.category,
                score=prior.score,
                triggers=prior.triggers + ["stage3_error"],
                stage=3,
            )

    def _parse_response(self, content: str) -> ClassificationResult:
        """Parse the model's JSON verdict, tolerating reasoning prose.

        Reasoning models emit chain-of-thought before the JSON, so a plain
        ``loads`` of the whole response fails. Fall back to extracting the
        last balanced ``{...}`` object from the text.
        """
        data = _extract_json_object(content)
        if data is None:
            logger.warning("Stage-3 returned no parseable JSON: %s", content[:200])
            return ClassificationResult(
                category="safe",
                score=0.0,
                triggers=["stage3_parse_error"],
                stage=3,
            )

        category = data.get("category", "safe")
        if category not in ("self_harm", "criminal_intent", "safe"):
            category = "safe"

        try:
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0

        # A dedicated Stage-3 safety classifier's category IS the verdict.
        # Its self-reported score is confidence and is often moderate even
        # for clear harm, so a harmful category is floored to a confident
        # value rather than being re-thresholded away downstream.
        if category == "self_harm":
            score = 1.0  # Critical override.
        elif category == "criminal_intent":
            score = max(score, 0.9)

        reasoning = data.get("reasoning", "")

        return ClassificationResult(
            category=category,
            score=score,
            triggers=[f"openai_chat:{category}"],
            stage=3,
            reasoning=reasoning,
        )