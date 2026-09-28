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

"""ONNX Runtime encoder for Stage 2 — no PyTorch required.

sentence-transformers pulls in the full PyTorch runtime (~2 GB installed),
whose dynamic-graph machinery is dead weight for single-message inference.
This encoder runs the pre-exported ``onnx/model.onnx`` that the model's
Hugging Face repository already ships, using only ``onnxruntime``,
``tokenizers``, and ``huggingface_hub``.

**Install:** ``pip install humane-proxy[onnx]``

Output parity: the encoder replicates the SentenceTransformer module
chain — tokenize (with the model's ``max_seq_length`` truncation), run
the transformer, mean-pool over the attention mask, and L2-normalize iff
the model declares a Normalize module — so embeddings match
``SentenceTransformer(name).encode()`` numerically.
"""

from __future__ import annotations

import logging
from typing import Any

from humane_proxy._json import loads as _json_loads

logger = logging.getLogger("humane_proxy.classifiers.onnx_encoder")

# numpy is a core dependency — only the ONNX stack is optional.
import numpy as np

try:
    import onnxruntime as _ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    _ort = None  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]
    Tokenizer = None  # type: ignore[assignment,misc]

_DEFAULT_MAX_SEQ_LENGTH = 512


def _resolve_repo_id(model_name: str) -> str:
    """Resolve a bare model name the way sentence-transformers does."""
    if "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def _mean_pool(last_hidden: Any, attention_mask: Any) -> Any:
    """Mean-pool token embeddings, ignoring padding positions."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


class OnnxEncoder:
    """Drop-in replacement for ``SentenceTransformer.encode()``.

    Downloads the repo's pre-exported ONNX graph and tokenizer on first
    construction (cached by huggingface_hub thereafter).
    """

    def __init__(self, model_name: str) -> None:
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "ONNX encoder requires onnxruntime, tokenizers, and "
                "huggingface_hub. Install with: pip install humane-proxy[onnx]"
            )

        repo_id = _resolve_repo_id(model_name)
        model_path = hf_hub_download(repo_id, "onnx/model.onnx")
        tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")

        max_seq = _DEFAULT_MAX_SEQ_LENGTH
        try:
            sbert_cfg_path = hf_hub_download(repo_id, "sentence_bert_config.json")
            with open(sbert_cfg_path, "r", encoding="utf-8") as fh:
                max_seq = int(_json_loads(fh.read()).get(
                    "max_seq_length", _DEFAULT_MAX_SEQ_LENGTH
                ))
        except Exception:
            logger.debug(
                "%s has no sentence_bert_config.json; using max_seq_length=%d",
                repo_id, max_seq,
            )

        # Replicate the model's module chain: normalize only if the repo
        # declares a Normalize module (all-MiniLM-L6-v2 does).
        self._normalize = False
        try:
            modules_path = hf_hub_download(repo_id, "modules.json")
            with open(modules_path, "r", encoding="utf-8") as fh:
                modules = _json_loads(fh.read())
            self._normalize = any(
                "Normalize" in str(m.get("type", "")) for m in modules
            )
        except Exception:
            logger.debug("%s has no modules.json; skipping normalization", repo_id)

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_seq)
        pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        self._session = _ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        logger.info(
            "ONNX Stage-2 encoder ready: %s (max_seq=%d, normalize=%s)",
            repo_id, max_seq, self._normalize,
        )

    def encode(self, sentences: list[str], show_progress_bar: bool = False) -> Any:
        """Encode *sentences* to a 2D numpy array of embeddings.

        Signature mirrors ``SentenceTransformer.encode()`` so the encoder
        slots into the existing model cache unchanged.
        """
        encodings = self._tokenizer.encode_batch(list(sentences))
        feed: dict[str, Any] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            ),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )

        last_hidden = self._session.run(None, feed)[0]
        embeddings = _mean_pool(last_hidden, feed["attention_mask"])

        if self._normalize:
            norms = np.clip(
                np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9, None
            )
            embeddings = embeddings / norms

        return embeddings
