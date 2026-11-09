import json
import xml.etree.ElementTree as ET

import pytest

from ai_stylist.catalog import load_catalog
from ai_stylist.index import load_or_build
from ai_stylist.thumbnails import COLORS, GLYPHS, render


def test_catalog_is_valid_and_sized(products):
    assert 60 <= len(products) <= 80
    assert {p["brand"] for p in products} == {"Atelier Nove"}


def test_invalid_record_is_rejected(cfg, tmp_path):
    bad = tmp_path / "catalog.json"
    bad.write_text(json.dumps([{"sku": "bad"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_catalog(bad, cfg["paths"]["schema"])


def test_every_color_and_category_has_a_thumbnail_style(products):
    assert {c for p in products for c in p["colors"]} <= set(COLORS)
    assert {p["category"] for p in products} <= set(GLYPHS)


def test_thumbnail_is_valid_svg_with_product_colors(products):
    p = next(p for p in products if len(p["colors"]) > 1)
    svg = render(p)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    for c in p["colors"]:
        assert COLORS[c] in svg
    assert p["sku"] in svg


def test_committed_thumbnails_are_up_to_date(cfg, products):
    for p in products:
        path = cfg["paths"]["thumbnails"] / f"{p['sku']}.svg"
        assert path.read_text(encoding="utf-8") == render(p), (
            f"run python -m ai_stylist.thumbnails ({p['sku']})"
        )
        assert p["image"] == f"data/thumbnails/{p['sku']}.svg"


def test_index_is_rebuilt_only_when_catalog_changes(cfg, products, encoder, tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(cfg["paths"]["catalog"].read_bytes())
    local = {**cfg, "paths": {**cfg["paths"], "catalog": catalog, "cache": tmp_path / "cache"}}
    before = encoder.calls
    load_or_build(local, products, encoder)
    load_or_build(local, products, encoder)
    assert encoder.calls == before + 1  # second call loads from disk
    catalog.write_text(catalog.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    load_or_build(local, products, encoder)
    assert encoder.calls == before + 2  # hash changed -> rebuilt
