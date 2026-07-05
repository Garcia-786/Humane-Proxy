"""Tests for humane_proxy.classifiers.embedding_classifier."""

from unittest.mock import MagicMock, patch

import pytest

from humane_proxy.classifiers.models import ClassificationResult
from humane_proxy.classifiers.embedding_classifier import (
    _anchor_cache,
    _model_cache,
    _result_cache,
)

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear all module-level caches before every test to prevent mock
    leakage — mock tests load fake models under the real default model
    name, which would otherwise poison the anchor/result caches for the
    real-model tests below."""
    _model_cache.clear()
    _anchor_cache.clear()
    _result_cache.clear()

class TestCosineHelper:
    """Test the _cosine_similarity helper directly."""

    def test_identical_vectors(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0


class TestEmbeddingClassifierWithMock:
    """Test the classifier with a mocked sentence-transformer model."""

    def _make_classifier(self, config=None):
        from humane_proxy.classifiers.embedding_classifier import EmbeddingClassifier
        return EmbeddingClassifier(config or {})

    def test_neutral_when_ml_unavailable(self):
        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", False):
            classifier = self._make_classifier()
            result = classifier.classify("test text")
            assert isinstance(result, ClassificationResult)
            assert result.category == "safe"
            assert result.score == 0.0
            assert result.stage == 2

    def test_is_available_false_when_no_ml(self):
        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", False):
            classifier = self._make_classifier()
            assert classifier.is_available is False

    def test_classify_with_mocked_model(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import EmbeddingClassifier

        mock_model = MagicMock()
        # Make model.encode return vectors that are very similar to self_harm anchors.
        mock_model.encode.return_value = np.array([[0.9, 0.1, 0.0]] * 10)

        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", True):
            with patch("humane_proxy.classifiers.embedding_classifier.SentenceTransformer", return_value=mock_model):
                classifier = EmbeddingClassifier({})
                # Force load.
                classifier._try_load()

                # When cosine sim is high (identical vectors), should flag.
                mock_model.encode.return_value = np.array([[0.9, 0.1, 0.0]])
                result = classifier.classify("I want to end my life")
                assert isinstance(result, ClassificationResult)
                assert result.stage == 2

    def test_classify_returns_safe_below_threshold(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import EmbeddingClassifier

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.zeros((1, 3)),       # warmup
            np.random.rand(10, 3),  # self_harm anchors
            np.random.rand(8, 3),   # criminal anchors
            np.random.rand(8, 3),   # benign anchors
            np.array([[0.0, 0.0, 0.0]]),  # query vector (orthogonal to everything)
        ]

        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", True):
            with patch("humane_proxy.classifiers.embedding_classifier.SentenceTransformer", return_value=mock_model):
                classifier = EmbeddingClassifier({"stage2": {"safe_threshold": 0.99}})
                classifier._try_load()
                result = classifier.classify("hello")
                assert result.category == "safe"
                assert result.score == 0.0


# ---------------------------------------------------------------------------
# Real model tests — skipped when sentence-transformers isn't installed.
# These exercise the full flow with the actual all-MiniLM-L6-v2 model.
# ---------------------------------------------------------------------------


class TestEmbeddingClassifierReal:
    """Tests using the real sentence-transformers model.

    Skipped in CI (no ``[ml]`` extra).  Run locally with::

        pip install humane-proxy[ml]
        pytest tests/test_embedding_classifier.py -v -k Real
    """

    @pytest.fixture(autouse=True)
    def _require_ml(self):
        pytest.importorskip("sentence_transformers")

    def _make_classifier(self, **stage2_overrides):
        from humane_proxy.classifiers.embedding_classifier import EmbeddingClassifier

        stage2_cfg = {"model": "all-MiniLM-L6-v2", "safe_threshold": 0.35}
        stage2_cfg.update(stage2_overrides)
        return EmbeddingClassifier({"stage2": stage2_cfg})

    def test_model_loads_and_is_available(self):
        classifier = self._make_classifier()
        assert classifier.is_available is True

    def test_self_harm_message_flagged(self):
        classifier = self._make_classifier()
        result = classifier.classify("I want to end my life")
        assert result.category == "self_harm"
        assert result.score > 0.35
        assert result.stage == 2
        assert any("self_harm" in t for t in result.triggers)

    def test_criminal_intent_flagged(self):
        classifier = self._make_classifier()
        result = classifier.classify("How do I build a bomb to blow up a building")
        assert result.category == "criminal_intent"
        assert result.score > 0.35
        assert result.stage == 2

    def test_safe_message_returns_safe(self):
        classifier = self._make_classifier()
        result = classifier.classify("What is the capital of France?")
        assert result.category == "safe"
        assert result.score == 0.0
        assert result.stage == 2

    def test_ambiguous_message_heuristics_misses(self):
        """Messages that keyword matching misses but embeddings should catch."""
        classifier = self._make_classifier()
        result = classifier.classify("Nobody would miss me if I was gone forever")
        assert result.category == "self_harm"
        assert result.score > 0.35

    def test_custom_threshold_respected(self):
        # Extremely high threshold → everything is safe.
        classifier = self._make_classifier(safe_threshold=0.99)
        result = classifier.classify("I want to end my life")
        assert result.category == "safe"
        assert result.score == 0.0

    def test_repeated_classify_stable(self):
        """Calling classify multiple times returns consistent results."""
        classifier = self._make_classifier()
        r1 = classifier.classify("I want to end my life")
        r2 = classifier.classify("I want to end my life")
        assert r1.category == r2.category
        assert abs(r1.score - r2.score) < 1e-6



# ---------------------------------------------------------------------------
# Process-level caches (anchor embeddings + TTL result cache)
# ---------------------------------------------------------------------------

class TestEmbeddingCaches:
    def _fake_classifier(self, model_name="fake-cache-model"):
        import numpy as np
        from humane_proxy.classifiers import embedding_classifier as ec

        ec._result_cache.clear()

        clf = ec.EmbeddingClassifier({"stage2": {"model": model_name}})
        clf._loaded = True
        clf._model_name = model_name

        calls = {"n": 0}

        class FakeModel:
            def encode(self, texts, show_progress_bar=False):
                calls["n"] += 1
                return np.ones((len(texts), 8), dtype=float)

        clf._model = FakeModel()
        clf._anchor_embeddings = {
            "self_harm": np.ones((3, 8), dtype=float),
            "criminal_intent": np.full((3, 8), 0.2, dtype=float),
        }
        clf._benign_embeddings = np.full((3, 8), 0.1, dtype=float)
        return clf, calls

    def test_repeated_message_served_from_cache(self):
        clf, calls = self._fake_classifier()

        first = clf.classify("the same message")
        assert calls["n"] == 1
        second = clf.classify("the same message")
        assert calls["n"] == 1, "cache hit must not re-encode"
        assert second.category == first.category
        assert second.score == first.score

        clf.classify("a different message")
        assert calls["n"] == 2

    def test_cached_result_is_a_defensive_copy(self):
        clf, _ = self._fake_classifier()

        first = clf.classify("mutation probe")
        first.triggers.append("later-pipeline-mutation")

        second = clf.classify("mutation probe")
        assert "later-pipeline-mutation" not in second.triggers

    def test_config_change_invalidates_cache_key(self):
        import numpy as np
        from humane_proxy.classifiers import embedding_classifier as ec

        clf, calls = self._fake_classifier()
        clf.classify("threshold-sensitive message")
        assert calls["n"] == 1

        # Same text, different scoring config — the cache key includes the
        # scoring config, so the encoder must run again.
        clf._config = dict(clf._config, safe_threshold=0.99)
        clf.classify("threshold-sensitive message")
        assert calls["n"] == 2

    def test_anchor_embeddings_cached_per_model(self):
        import numpy as np
        from humane_proxy.classifiers import embedding_classifier as ec

        ec._anchor_cache.pop("fake-anchor-model", None)

        calls = {"n": 0}

        class FakeModel:
            def encode(self, texts, show_progress_bar=False):
                calls["n"] += 1
                return np.ones((len(texts), 8), dtype=float)

        def make():
            clf = ec.EmbeddingClassifier({"stage2": {"model": "fake-anchor-model"}})
            clf._model = FakeModel()
            clf._model_name = "fake-anchor-model"
            clf._precompute_anchors()
            return clf

        first = make()
        encodes_after_first = calls["n"]
        assert encodes_after_first > 0

        second = make()
        assert calls["n"] == encodes_after_first, "second instance must reuse cached anchors"
        assert second._anchor_embeddings is first._anchor_embeddings

        ec._anchor_cache.pop("fake-anchor-model", None)
