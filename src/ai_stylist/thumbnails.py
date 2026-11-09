"""Generate SVG thumbnails from product attributes: category glyph in the main color, color swatches, label.

No photos, so no licensing issues. Run: python -m ai_stylist.thumbnails
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from .catalog import load_catalog
from .config import load_config

INK, IVORY = "#1F1B24", "#F6F1E9"
COLORS = {
    "beige": "#D8C3A5",
    "black": "#1F1B24",
    "blue": "#3D6FB6",
    "blush": "#E8B4B0",
    "brown": "#7A4E2D",
    "burgundy": "#7B2335",
    "camel": "#C19A6B",
    "charcoal": "#3A3A3C",
    "gold": "#C9A642",
    "green": "#4F7A4A",
    "grey": "#9A9A9E",
    "indigo": "#2E3A63",
    "ivory": "#FFFDF5",
    "light blue": "#A9C8E8",
    "mid blue": "#5B7FA8",
    "navy": "#1F2F4D",
    "olive": "#6B6B3A",
    "red": "#B23A3A",
    "sage": "#6B8F71",
    "terracotta": "#C8553D",
    "white": "#FFFFFF",
    "yellow": "#E9C46A",
}
# Simple silhouettes drawn in a 120x120 box.
GLYPHS = {
    "top": "M30,22 L48,14 Q60,24 72,14 L90,22 L104,44 L88,52 L88,106 L32,106 L32,52 L16,44 Z",
    "shirt": "M30,20 L50,12 L60,26 L70,12 L90,20 L106,58 L90,64 L88,108 L32,108 L30,64 L14,58 Z M60,26 L60,108",
    "knitwear": "M28,24 L46,14 Q60,22 74,14 L92,24 L108,96 L92,100 L88,62 L88,104 L32,104 L32,62 L28,100 L12,96 Z",
    "dress": "M46,10 L74,10 L72,38 Q86,46 98,110 L22,110 Q34,46 48,38 Z",
    "skirt": "M38,18 L82,18 L84,30 L102,104 L18,104 L36,30 Z",
    "trousers": "M36,12 L84,12 L92,108 L68,108 L60,44 L52,108 L28,108 Z",
    "jeans": "M36,12 L84,12 L90,108 L68,108 L60,46 L52,108 L30,108 Z M36,24 L84,24",
    "outerwear": "M36,14 L60,30 L84,14 L104,30 L110,104 L90,104 L88,60 L88,110 L32,110 L32,60 L30,104 L10,104 L16,30 Z M60,30 L60,110",
    "shoes": "M14,60 L46,60 Q60,76 86,78 Q108,82 108,94 L108,100 L14,100 Z",
    "bag": "M24,44 L96,44 L104,106 L16,106 Z M42,44 Q42,14 60,14 Q78,14 78,44",
    "accessory": "M20,52 Q60,28 100,52 Q60,76 20,52 Z M60,52 L54,104 L66,104 Z",
}
W, H = 200, 240


def hex_for(color: str) -> str:
    """Hex value for a catalog color name; raises KeyError for unknown names so tests catch gaps."""
    return COLORS[color]


def render(p: dict) -> str:
    """Return the SVG markup for one product."""
    main = hex_for(p["colors"][0])
    swatches = "".join(
        f'<rect x="{16 + i * 26}" y="16" width="20" height="20" rx="10" fill="{hex_for(c)}" '
        f'stroke="{INK}" stroke-opacity="0.25"/>'
        for i, c in enumerate(p["colors"])
    )
    name = p["name"] if len(p["name"]) <= 24 else p["name"][:23] + "…"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'role="img" aria-label="{escape(p["name"])}, {escape(", ".join(p["colors"]))}">\n'
        f'  <rect width="{W}" height="{H}" rx="12" fill="{IVORY}"/>\n'
        f"  {swatches}\n"
        f'  <g transform="translate(40 44)"><path d="{GLYPHS[p["category"]]}" fill="{main}" '
        f'stroke="{INK}" stroke-opacity="0.35" stroke-width="2" stroke-linejoin="round"/></g>\n'
        f'  <text x="16" y="198" font-family="-apple-system, Segoe UI, Roboto, Arial, sans-serif" '
        f'font-size="14" font-weight="600" fill="{INK}">{escape(name)}</text>\n'
        f'  <text x="16" y="220" font-family="-apple-system, Segoe UI, Roboto, Arial, sans-serif" '
        f'font-size="12" fill="{INK}" fill-opacity="0.6">{p["sku"]} · {p["category"]}</text>\n'
        "</svg>\n"
    )


def write_all(products: list[dict], out_dir: Path) -> int:
    """Write one SVG per product (newline='\\n' so output is identical on every OS)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in products:
        (out_dir / f"{p['sku']}.svg").write_text(render(p), encoding="utf-8", newline="\n")
    return len(products)


def main() -> None:
    cfg = load_config()
    products = load_catalog(cfg["paths"]["catalog"], cfg["paths"]["schema"])
    n = write_all(products, cfg["paths"]["thumbnails"])
    print(f"{n} thumbnails -> {cfg['paths']['thumbnails']}")


if __name__ == "__main__":
    main()
