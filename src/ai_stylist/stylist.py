"""The stylist: retrieve, optionally generate, and enforce grounding in code.

Golden rule: the answer may only cite products that were retrieved for this question. Any other SKU
is removed from the text and reported. Names and prices shown to the shopper come from the catalog.
"""

from __future__ import annotations

import dataclasses
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import llm
from .index import Encoder
from .query import Query, parse_query
from .search import Hit, Look, complete_look, rank

SKU = re.compile(r"\b[A-Z]{2,5}-\d{4}\b")
_BRACKETS = re.compile(r"\[([^\[\]]*)\]")

SYSTEM_PROMPT = """You are the online stylist of Atelier Nove, a fashion store.
Answer the shopper's question using ONLY the products listed below. They were selected by the store's
search for this question; never mention any other product, brand or SKU.
Rules:
- Cite every product you mention with its SKU in square brackets, e.g. "the Linen Midi Dress [AN-0001]".
- Do not state prices, discounts, stock or delivery: the shop shows prices next to each item.
- Explain briefly why the pieces suit the occasion and work together (color, material, fit, season).
- If an outfit is given, present it as one look. Keep the answer to 3-5 sentences, friendly and specific.
- Reply in English. Return JSON: {"answer": "..."}"""


@dataclass
class Answer:
    question: str
    query: Query
    hits: list[Hit]
    look: Look | None
    mode: str  # "full" or "retrieval-only"
    text: str
    cited: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    note: str | None = None
    error: str | None = None
    retrieval_s: float = 0.0
    generation_s: float = 0.0


def ground(text: str, allowed: set[str]) -> tuple[str, list[str], list[str]]:
    """Remove every SKU not in `allowed`. Returns (clean text, cited SKUs, removed SKUs), in order."""
    cited: list[str] = []
    removed: list[str] = []

    def note(sku: str) -> bool:
        target = cited if sku in allowed else removed
        if sku not in target:
            target.append(sku)
        return sku in allowed

    def fix_brackets(m: re.Match) -> str:
        skus = SKU.findall(m.group(1))
        if not skus:
            return m.group(0)
        kept = [s for s in skus if note(s)]
        return f"[{', '.join(kept)}]" if kept else ""

    text = _BRACKETS.sub(fix_brackets, text)
    text = SKU.sub(lambda m: m.group(0) if note(m.group(0)) else "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return text.strip(), cited, removed


def _product_line(p: dict) -> str:
    return (
        f"[{p['sku']}] {p['name']}: {p['gender']} {p['category']}; colors {', '.join(p['colors'])}; "
        f"materials {', '.join(p['materials'])}; {p['pattern']}, {p['fit']} fit; "
        f"occasions {', '.join(p['occasions'])}; season {p['season']}. {p['description'] or ''}"
    )


def build_messages(question: str, hits: list[Hit], look: Look | None) -> list[dict]:
    """Chat messages containing the retrieved products only."""
    if look:
        context = "Outfit selected by the store (one look):\n" + "\n".join(
            _product_line(h.product) for h in look.hits
        )
    else:
        context = "Products selected by the store:\n" + "\n".join(
            _product_line(h.product) for h in hits
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nShopper's question: {question}"},
    ]


def _constraints(q: Query) -> str:
    """E.g. 'work, AW (men)'."""
    parts = [*q.occasions, *([q.season] if q.season else []), *q.categories]
    return ", ".join(parts) + (f" ({q.gender})" if q.gender else "")


def template_answer(q: Query, hits: list[Hit], look: Look | None) -> str:
    """Explanation without an LLM (retrieval-only mode). Names and prices come from the catalog."""
    if look:
        items = "; ".join(f"{h.product['name']} [{h.product['sku']}]" for h in look.hits)
        budget = f", within your €{q.budget:g} budget" if q.budget is not None else ""
        return (
            f"A complete look for {_constraints(q)}{budget}: {items}. "
            f"Total €{look.total:g}. The tags under each card show which of your constraints it matches."
        )
    items = "; ".join(f"{h.product['name']} [{h.product['sku']}]" for h in hits)
    which = f" ({_constraints(q)})" if _constraints(q) else ""
    return f"Best matches in the catalog{which}: {items}."


def retrieve(
    question: str,
    products: list[dict],
    vectors: np.ndarray,
    encoder: Encoder,
    vocab: dict,
    top_k: int = 6,
    look_candidates: int = 5,
    overrides: dict | None = None,
) -> tuple[Query, list[Hit], Look | None, str | None]:
    """Parse, filter, rank and (for occasion questions) complete the look. Returns (query, hits, look, note)."""
    q = parse_query(question, vocab)
    if overrides:
        q = dataclasses.replace(q, **{k: v for k, v in overrides.items() if v not in (None, [])})
    qv = encoder.encode([question])[0]
    note = None
    look = complete_look(products, vectors, qv, q, look_candidates) if q.wants_look else None
    if q.wants_look and look is None:
        budget = f" within €{q.budget:g}" if q.budget is not None else ""
        note = f"No complete outfit fits{budget} in this catalog; showing the closest individual items."
    hits = look.hits if look else rank(products, vectors, qv, q)[:top_k]
    if not hits:
        note = "Nothing in the catalog meets all these constraints. Try a higher budget or fewer filters."
    return q, hits, look, note


def answer(
    question: str,
    cfg: dict,
    products: list[dict],
    vectors: np.ndarray,
    encoder: Encoder,
    vocab: dict,
    use_llm: bool,
    overrides: dict | None = None,
    chat: Callable[[list[dict], dict], str] = llm.chat,
) -> Answer:
    """Answer a shopper's question; falls back to the template if the LLM is off or fails."""
    r = cfg["retrieval"]
    t0 = time.perf_counter()
    q, hits, look, note = retrieve(
        question, products, vectors, encoder, vocab, r["top_k"], r["look_candidates"], overrides
    )
    retrieval_s = time.perf_counter() - t0
    result = Answer(
        question, q, hits, look, "retrieval-only", "", note=note, retrieval_s=retrieval_s
    )
    if not hits:
        result.text = note or ""
        return result
    if use_llm:
        t1 = time.perf_counter()
        try:
            raw = chat(build_messages(question, hits, look), cfg)
            allowed = {h.product["sku"] for h in hits}
            result.text, result.cited, result.removed = ground(raw, allowed)
            result.mode = "full"
        except llm.LLMError as e:
            result.error = str(e)
        result.generation_s = time.perf_counter() - t1
    if result.mode != "full":
        result.text = template_answer(q, hits, look)
        result.cited = [h.product["sku"] for h in hits]
    return result
