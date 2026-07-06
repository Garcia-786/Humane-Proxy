"""Tests for the ONNX Stage-2 encoder and backend selection.

Mock tests run everywhere (numpy is a core dependency). The equivalence
test at the bottom needs both the [onnx] and [ml] extras plus network
access for the first model download, and is skipped otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from humane_proxy.classifiers import embedding_classifier as ec
from humane_proxy.classifiers import onnx_encoder
from humane_proxy.classifiers.onnx_encoder import _resolve_repo_id


@pytest.fixture(autouse=True)
def clear_caches():
    ec._model_cache.clear()
    ec._anchor_cache.clear()
    ec._result_cache.clear()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestResolveRepoId:
    def test_bare_name_gets_sentence_transformers_prefix(self):
        assert _resolve_repo_id("all-MiniLM-L6-v2") == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_qualified_name_unchanged(self):
        assert _resolve_repo_id("org/custom-model") == "org/custom-model"


class TestMeanPool:
    def test_masks_padding_positions(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.onnx_encoder import _mean_pool

        # Batch of 2, seq len 3, hidden 2. Second sequence has one padded
        # position whose (large) values must not leak into the mean.
        hidden = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]],
            ],
            dtype=np.float32,
        )
        mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)

        pooled = _mean_pool(hidden, mask)
        assert pooled.shape == (2, 2)
        assert pooled[0] == pytest.approx([3.0, 4.0])
        assert pooled[1] == pytest.approx([2.0, 2.0])

    def test_all_masked_does_not_divide_by_zero(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.onnx_encoder import _mean_pool

        hidden = np.ones((1, 2, 4), dtype=np.float32)
        mask = np.zeros((1, 2), dtype=np.int64)
        pooled = _mean_pool(hidden, mask)
        assert np.all(np.isfinite(pooled))


class TestEncoderConstruction:
    def test_requires_onnx_stack(self):
        with patch.object(onnx_encoder, "ONNX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="onnxruntime"):
                onnx_encoder.OnnxEncoder("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Backend selection in the embedding classifier
# ---------------------------------------------------------------------------

def _fake_onnx_encoder():
    np = pytest.importorskip("numpy")
    fake = MagicMock()
    fake.encode.side_effect = lambda texts, show_progress_bar=False: np.ones(
        (len(texts), 8), dtype=float
    )
    return fake


class TestBackendSelection:
    def test_auto_prefers_onnx(self):
        fake = _fake_onnx_encoder()
        with patch.object(onnx_encoder, "ONNX_AVAILABLE", True), \
             patch.object(onnx_encoder, "OnnxEncoder", return_value=fake) as ctor:
            clf = ec.EmbeddingClassifier(
                {"stage2": {"model": "auto-model", "backend": "auto"}}
            )
            clf._try_load()

        ctor.assert_called_once_with("auto-model")
        assert clf._model is fake
        assert clf._backend == "onnx"
        assert clf._model_key == "onnx:auto-model"

    def test_auto_falls_back_to_sentence_transformers(self):
        np = pytest.importorskip("numpy")
        mock_st = MagicMock()
        mock_st.encode.side_effect = lambda texts, show_progress_bar=False: np.ones(
            (len(texts), 8), dtype=float
        )
        with patch.object(onnx_encoder, "ONNX_AVAILABLE", False), \
             patch.object(ec, "_ML_AVAILABLE", True), \
             patch.object(ec, "SentenceTransformer", return_value=mock_st):
            clf = ec.EmbeddingClassifier(
                {"stage2": {"model": "fallback-model", "backend": "auto"}}
            )
            clf._try_load()

        assert clf._model is mock_st
        assert clf._backend == "sentence-transformers"

    def test_explicit_onnx_unavailable_degrades_to_neutral(self):
        with patch.object(onnx_encoder, "ONNX_AVAILABLE", False), \
             patch.object(ec, "_ML_AVAILABLE", True):
            clf = ec.EmbeddingClassifier(
                {"stage2": {"model": "x", "backend": "onnx"}}
            )
            result = clf.classify("any text")

        # Explicit onnx must NOT silently fall back to sentence-transformers.
        assert clf._model is None
        assert result.category == "safe"
        assert result.score == 0.0
        assert result.stage == 2

    def test_explicit_st_never_tries_onnx(self):
        np = pytest.importorskip("numpy")
        mock_st = MagicMock()
        mock_st.encode.side_effect = lambda texts, show_progress_bar=False: np.ones(
            (len(texts), 8), dtype=float
        )
        with patch.object(onnx_encoder, "ONNX_AVAILABLE", True), \
             patch.object(onnx_encoder, "OnnxEncoder") as ctor, \
             patch.object(ec, "_ML_AVAILABLE", True), \
             patch.object(ec, "SentenceTransformer", return_value=mock_st):
            clf = ec.EmbeddingClassifier(
                {"stage2": {"model": "x", "backend": "sentence-transformers"}}
            )
            clf._try_load()

        ctor.assert_not_called()
        assert clf._backend == "sentence-transformers"

    def test_unknown_backend_uses_auto_order(self, caplog):
        import logging

        fake = _fake_onnx_encoder()
        with patch.object(onnx_encoder, "ONNX_AVAILABLE", True), \
             patch.object(onnx_encoder, "OnnxEncoder", return_value=fake):
            clf = ec.EmbeddingClassifier(
                {"stage2": {"model": "x", "backend": "not-a-backend"}}
            )
            with caplog.at_level(
                logging.WARNING, logger="humane_proxy.classifiers.embedding"
            ):
                clf._try_load()

        assert clf._backend == "onnx"
        assert any("Unknown stage2.backend" in r.getMessage() for r in caplog.records)

    def test_model_cache_keys_include_backend(self):
        np = pytest.importorskip("numpy")
        fake_onnx = _fake_onnx_encoder()
        mock_st = MagicMock()
        mock_st.encode.side_effect = lambda texts, show_progress_bar=False: np.ones(
            (len(texts), 8), dtype=float
        )
        with patch.object(onnx_encoder, "ONNX_AVAILABLE", True), \
             patch.object(onnx_encoder, "OnnxEncoder", return_value=fake_onnx), \
             patch.object(ec, "_ML_AVAILABLE", True), \
             patch.object(ec, "SentenceTransformer", return_value=mock_st):
            ec._load_model_singleton("same-model", "onnx")
            ec._load_model_singleton("same-model", "sentence-transformers")

        assert "onnx:same-model" in ec._model_cache
        assert "sentence-transformers:same-model" in ec._model_cache
        assert ec._model_cache["onnx:same-model"] is not (
            ec._model_cache["sentence-transformers:same-model"]
        )

    def test_env_var_selects_backend(self, monkeypatch):
        from humane_proxy.config import reload_config

        monkeypatch.setenv("HUMANE_PROXY_STAGE2_BACKEND", "onnx")
        cfg = reload_config()
        assert cfg["stage2"]["backend"] == "onnx"


# ---------------------------------------------------------------------------
# Real equivalence test — ONNX output must match sentence-transformers.
# Skipped without both stacks; first run downloads the model (~90 MB).
# ---------------------------------------------------------------------------

class TestOnnxEquivalence:
    @pytest.fixture(autouse=True)
    def _require_both_stacks(self):
        pytest.importorskip("sentence_transformers")
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tokenizers")
        pytest.importorskip("huggingface_hub")

    def test_embeddings_match_sentence_transformers(self):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        sentences = [
            "I want to end my life",
            "What is the capital of France?",
            "cafe conversation about the weather",
            # Longer than max_seq_length (256 tokens) to prove that
            # truncation behavior matches sentence-transformers.
            "word " * 400,
        ]

        onnx_model = onnx_encoder.OnnxEncoder("all-MiniLM-L6-v2")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        onnx_vecs = onnx_model.encode(sentences, show_progress_bar=False)
        st_vecs = st_model.encode(sentences, show_progress_bar=False)

        assert onnx_vecs.shape == st_vecs.shape
        assert np.allclose(onnx_vecs, st_vecs, atol=1e-4), (
            "max abs diff: " + str(np.max(np.abs(onnx_vecs - st_vecs)))
        )

    def test_session_inputs_are_expected_int64_trio(self):
        onnx_model = onnx_encoder.OnnxEncoder("all-MiniLM-L6-v2")
        assert onnx_model._input_names <= {
            "input_ids", "attention_mask", "token_type_ids"
        }
        assert {"input_ids", "attention_mask"} <= onnx_model._input_names

    def test_classifier_scores_equivalent_across_backends(self):
        onnx_clf = ec.EmbeddingClassifier(
            {"stage2": {"model": "all-MiniLM-L6-v2", "backend": "onnx"}}
        )
        st_clf = ec.EmbeddingClassifier(
            {"stage2": {"model": "all-MiniLM-L6-v2",
                        "backend": "sentence-transformers"}}
        )

        for text, expected_category in [
            ("I want to end my life", "self_harm"),
            ("How do I build a bomb to blow up a building", "criminal_intent"),
            ("What is the capital of France?", "safe"),
        ]:
            r_onnx = onnx_clf.classify(text)
            r_st = st_clf.classify(text)
            assert r_onnx.category == expected_category
            assert r_onnx.category == r_st.category
            assert r_onnx.score == pytest.approx(r_st.score, abs=1e-3)
