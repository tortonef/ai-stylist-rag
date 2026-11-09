"""Smoke check run at the end of setup: catalog valid, thumbnails present, the real index answers the
SPEC example with a complete look within budget, and Ollama is reachable (or retrieval-only mode)."""

from __future__ import annotations

from .catalog import load_catalog
from .config import load_config
from .embed import load_encoder
from .index import load_or_build
from .llm import unavailable_reason
from .query import load_vocabulary
from .stylist import retrieve

EXAMPLE = "What should I wear to a country wedding in May, under €300?"


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    products = load_catalog(paths["catalog"], paths["schema"])
    missing = [p["sku"] for p in products if not (paths["thumbnails"] / f"{p['sku']}.svg").exists()]
    if missing:
        raise SystemExit(f"Smoke check failed: missing thumbnails {missing[:5]}")
    encoder = load_encoder(cfg)
    vectors = load_or_build(cfg, products, encoder)
    _, _, look, _ = retrieve(
        EXAMPLE, products, vectors, encoder, load_vocabulary(paths["vocabulary"])
    )
    if look is None or look.total > 300:
        raise SystemExit(
            "Smoke check failed: no complete look within budget for the example question."
        )
    print(
        f"Smoke check OK: {len(products)} products, index {vectors.shape}, "
        f"example look of {len(look.hits)} items for EUR {look.total:g}."
    )
    reason = unavailable_reason(cfg)
    print(
        "Full mode: ready."
        if reason is None
        else f"Full mode off ({reason}) -> retrieval-only mode."
    )


if __name__ == "__main__":
    main()
