"""Vector index: a NumPy matrix on disk, rebuilt automatically when the catalog or the model changes.

Run by setup (downloads the model, builds the index): python -m ai_stylist.index
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .catalog import catalog_hash, load_catalog, product_text
from .config import load_config


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _files(cache: Path) -> tuple[Path, Path]:
    d = cache / "index"
    return d / "embeddings.npy", d / "meta.json"


def _fingerprint(cfg: dict, products: list[dict]) -> dict:
    return {
        "catalog_sha256": catalog_hash(cfg["paths"]["catalog"]),
        "model": f"{cfg['embedding']['id']}@{cfg['embedding']['revision']}",
        "skus": [p["sku"] for p in products],
    }


def load_or_build(cfg: dict, products: list[dict], encoder: Encoder) -> np.ndarray:
    """Return one unit vector per product (same order as `products`), rebuilding if stale."""
    emb_path, meta_path = _files(cfg["paths"]["cache"])
    expected = _fingerprint(cfg, products)
    if emb_path.exists() and meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            if json.load(f) == expected:
                return np.load(emb_path)
    vectors = encoder.encode([product_text(p) for p in products])
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, vectors)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)
    return vectors


def main() -> None:
    from .embed import download, load_encoder, model_cache

    cfg = load_config()
    e = cfg["embedding"]
    path = download(e["id"], e["revision"], model_cache(cfg))
    print(f"Model ready: {e['id']} @ {e['revision'][:7]} in {path}")
    products = load_catalog(cfg["paths"]["catalog"], cfg["paths"]["schema"])
    vectors = load_or_build(cfg, products, load_encoder(cfg))
    print(f"Index ready: {vectors.shape[0]} products x {vectors.shape[1]} dimensions")


if __name__ == "__main__":
    main()
