"""Tests for Stage-3 providers (llamaguard, openai_moderation, openai_chat).

All tests mock HTTP calls — no real API keys needed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from humane_proxy.classifiers.models import ClassificationResult


# -----------------------------------------------------------------------
# LlamaGuard
# -----------------------------------------------------------------------

class TestLlamaGuard:
    def _make(self, config=None):
        from humane_proxy.classifiers.stage3.llamaguard import LlamaGuardClassifier
        return LlamaGuardClassifier(config or {})

    @pytest.mark.asyncio
    async def test_safe_verdict(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "safe"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("hello", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_unsafe_self_harm(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "unsafe\nS11"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("I want to die", ClassificationResult())
            assert result.category == "self_harm"
            assert result.score == 1.0
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_unsafe_criminal(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "unsafe\nS1"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("bomb instructions", ClassificationResult())
            assert result.category == "criminal_intent"
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_api_error_graceful(self):
        cls = self._make()
        prior = ClassificationResult(category="safe", score=0.0, triggers=["t1"])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("timeout")):
            result = await cls.classify("test", prior)
            assert "stage3_error" in result.triggers
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_multiple_category_codes(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "unsafe\nS1,S11"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("text", ClassificationResult())
            # S11 = self_harm takes priority over S1 = criminal
            assert result.category == "self_harm"


# -----------------------------------------------------------------------
# OpenAI Moderation
# -----------------------------------------------------------------------

class TestOpenAIModeration:
    def _make(self, config=None):
        from humane_proxy.classifiers.stage3.openai_moderation import OpenAIModerationClassifier
        return OpenAIModerationClassifier(config or {})

    @pytest.mark.asyncio
    async def test_not_flagged(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"flagged": False, "categories": {}, "category_scores": {}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("hello", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_self_harm_flagged(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"self-harm": True, "self-harm/intent": True},
            "category_scores": {"self-harm": 0.95, "self-harm/intent": 0.9},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "self_harm"
            assert result.score == 1.0  # critical override

    @pytest.mark.asyncio
    async def test_violence_flagged(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"violence": True},
            "category_scores": {"violence": 0.8},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "criminal_intent"

    @pytest.mark.asyncio
    async def test_flag_escalates_despite_low_raw_score(self):
        """A moderation flag is a calibrated decision; the raw category_score
        is often < 0.5 even when flagged. We must escalate on the flag, not
        pass the low raw score through to be re-thresholded away."""
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"violence": True},
            "category_scores": {"violence": 0.42},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("break someone's jaw", ClassificationResult())
            assert result.category == "criminal_intent"
            assert result.score >= 0.9  # confident detection, not 0.42

    @pytest.mark.asyncio
    async def test_illicit_category_maps_to_criminal(self):
        """omni-moderation's illicit categories (drugs, weapons, fraud) must
        map to criminal_intent — they were previously ignored."""
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"illicit": True},
            "category_scores": {"illicit": 0.3},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("how to run a pyramid scheme", ClassificationResult())
            assert result.category == "criminal_intent"
            assert result.score >= 0.9

    @pytest.mark.asyncio
    async def test_omni_model_sent_in_payload(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"flagged": False, "categories": {}, "category_scores": {}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await cls.classify("hello", ClassificationResult())
            sent = mock_post.call_args.kwargs["json"]
            assert sent["model"] == "omni-moderation-latest"

    @pytest.mark.asyncio
    async def test_harassment_stays_safe(self):
        """Harassment flags are noted but kept as 'safe' — not our domain."""
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"harassment": True},
            "category_scores": {"harassment": 0.7},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "safe"

    @pytest.mark.asyncio
    async def test_api_error_graceful(self):
        cls = self._make()
        prior = ClassificationResult(category="safe", score=0.1, triggers=["t1"])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("nope")):
            result = await cls.classify("test", prior)
            assert "stage3_error" in result.triggers


# -----------------------------------------------------------------------
# OpenAI Chat
# -----------------------------------------------------------------------

class TestOpenAIChat:
    def _make(self, config=None):
        from humane_proxy.classifiers.stage3.openai_chat import OpenAIChatClassifier
        return OpenAIChatClassifier(config or {})

    @pytest.mark.asyncio
    async def test_safe_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "safe", "score": 0.0, "reasoning": "Normal chat"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("hello", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_self_harm_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "self_harm", "score": 0.95, "reasoning": "Suicidal ideation"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("I want to die", ClassificationResult())
            assert result.category == "self_harm"
            assert result.score == 1.0  # critical override

    @pytest.mark.asyncio
    async def test_criminal_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "criminal_intent", "score": 0.8, "reasoning": "Violence"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("bomb", ClassificationResult())
            assert result.category == "criminal_intent"
            # Category authority: a harmful classification is floored so the
            # model's moderate self-reported score can't drop below threshold.
            assert result.score >= 0.9

    @pytest.mark.asyncio
    async def test_criminal_flag_floored_despite_low_score(self):
        """A confident criminal classification with a low self-reported score
        must still escalate (the classifier's category is the verdict)."""
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "criminal_intent", "score": 0.42, "reasoning": "x"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("detonate a bomb", ClassificationResult())
            assert result.category == "criminal_intent"
            assert result.score >= 0.9

    @pytest.mark.asyncio
    async def test_tolerant_parse_of_reasoning_prefixed_json(self):
        """Reasoning models emit chain-of-thought before the JSON; the parser
        must extract the trailing JSON object rather than failing."""
        cls = self._make()
        content = (
            "Let me think. The user is asking how to harm a person, which "
            "is a request for violence.\n\n"
            '{"category": "criminal_intent", "score": 0.95, "reasoning": "violence"}'
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "criminal_intent"
            assert "stage3_parse_error" not in result.triggers

    @pytest.mark.asyncio
    async def test_json_mode_off_by_default_and_max_tokens_configurable(self):
        cls = self._make({"stage3": {"openai_chat": {"max_tokens": 1500}}})
        assert cls._json_mode is False
        assert cls._max_tokens == 1500

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "safe", "score": 0.0}'}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mp:
            await cls.classify("hi", ClassificationResult())
            sent = mp.call_args.kwargs["json"]
            assert "response_format" not in sent
            assert sent["max_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_non_json_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "I cannot classify this"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "safe"
            assert "stage3_parse_error" in result.triggers

    @pytest.mark.asyncio
    async def test_unknown_category_defaults_safe(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "unknown_thing", "score": 0.5}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "safe"

    @pytest.mark.asyncio
    async def test_api_error_graceful(self):
        cls = self._make()
        prior = ClassificationResult(category="safe", triggers=["t1"])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("boom")):
            result = await cls.classify("test", prior)
            assert "stage3_error" in result.triggers


# -----------------------------------------------------------------------
# LlamaGuard category mapping regressions
# -----------------------------------------------------------------------

class TestLlamaGuardMapping:
    """Regressions for the S-code -> HumaneProxy category map."""

    def _make(self):
        from humane_proxy.classifiers.stage3.llamaguard import LlamaGuardClassifier
        return LlamaGuardClassifier({})

    def _mock_resp(self, content: str):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio
    async def test_s9_indiscriminate_weapons_is_criminal(self):
        """S9 (Indiscriminate Weapons) used to be mapped to 'safe'."""
        cls = self._make()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=self._mock_resp("unsafe\nS9")):
            result = await cls.classify("chemical weapon synthesis", ClassificationResult())
            assert result.category == "criminal_intent"
            assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_s10_hate_is_not_self_harm(self):
        """S10 (Hate) used to map to self_harm, sending hate speech a
        suicide-crisis care response with score forced to 1.0."""
        cls = self._make()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=self._mock_resp("unsafe\nS10")):
            result = await cls.classify("hateful message", ClassificationResult())
            assert result.category != "self_harm"
            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_unsafe_without_codes_scores_zero(self):
        """A bare 'unsafe' with no S-codes used to return safe with 0.85,
        silently inflating the combined pipeline score."""
        cls = self._make()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=self._mock_resp("unsafe")):
            result = await cls.classify("something", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0
            assert "llamaguard:unsafe_out_of_scope" in result.triggers

    @pytest.mark.asyncio
    async def test_out_of_scope_codes_score_zero(self):
        """Codes outside our domain (privacy/IP) must not inflate the score."""
        cls = self._make()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=self._mock_resp("unsafe\nS7,S8")):
            result = await cls.classify("privacy question", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0
