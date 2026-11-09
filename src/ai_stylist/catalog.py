"""Load and validate the product catalog; build the text that gets embedded."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

SEASON_WORDS = {"SS": "spring summer", "AW": "autumn winter", "all-season": "all season"}


def load_catalog(path: Path, schema_path: Path) -> list[dict]:
    """Return the products, raising ValueError if any record breaks the shared schema."""
    with path.open(encoding="utf-8") as f:
        products = json.load(f)
    with schema_path.open(encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft202012Validator(schema)
    seen: set[str] = set()
    for p in products:
        errors = sorted(validator.iter_errors(p), key=str)
        if errors:
            raise ValueError(f"{p.get('sku', '?')}: {errors[0].message}")
        if p["sku"] in seen:
            raise ValueError(f"Duplicate SKU {p['sku']}")
        seen.add(p["sku"])
    return products


def catalog_hash(path: Path) -> str:
    """SHA-256 of the catalog file: the index is rebuilt when it changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product_text(p: dict) -> str:
    """The text that represents a product for semantic search."""
    return (
        f"{p['name']}. {p['gender']} {p['category']}. "
        f"Colors: {', '.join(p['colors'])}. Materials: {', '.join(p['materials'])}. "
        f"{p['pattern']} pattern, {p['fit']} fit. Occasions: {', '.join(p['occasions'])}. "
        f"Season: {SEASON_WORDS[p['season']]}. {p['description'] or ''}"
    ).strip()
