"""all-MiniLM-L6-v2 sentence embeddings on CPU with plain transformers (mean pooling, as on the model card).

Downloads happen only in `download` (setup); the app loads the model offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Files transformers needs; skips ONNX/OpenVINO exports and duplicate weight formats.
ALLOW_PATTERNS = ["*.json", "vocab.txt", "model.safetensors"]


def download(model_id: str, revision: str, cache_dir: Path) -> Path:
    """Fetch the pinned model snapshot into the local cache (no-op if already there)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id, revision=revision, cache_dir=cache_dir, allow_patterns=ALLOW_PATTERNS
        )
    )


class MiniLMEncoder:
    """Encodes texts into L2-normalised vectors, so a dot product is the cosine similarity."""

    def __init__(self, model_id: str, revision: str, cache_dir: Path) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        kwargs = {"revision": revision, "cache_dir": cache_dir, "local_files_only": True}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            self.model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        except OSError as e:
            raise RuntimeError(
                f"Model '{model_id}' not found in {cache_dir}. Run the setup script first."
            ) from e
        self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), 384) float32 array of unit vectors."""
        with self._torch.inference_mode():
            batch = self.tokenizer(
                texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            tokens = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.numpy().astype(np.float32)


def model_cache(cfg: dict) -> Path:
    """Where the embedding model lives (gitignored)."""
    return cfg["paths"]["cache"] / "models"


def load_encoder(cfg: dict) -> MiniLMEncoder:
    """The configured encoder, from the local cache."""
    e = cfg["embedding"]
    return MiniLMEncoder(e["id"], e["revision"], model_cache(cfg))
