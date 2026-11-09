"""Hybrid search: deterministic hard filters first, then semantic ranking; plus complete the look."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as cartesian

import numpy as np

from .query import Query

# Outfit slots. "Base" is either one dress or a top plus a bottom.
TOPS = {"top", "shirt", "knitwear"}
BOTTOMS = {"skirt", "trousers", "jeans"}
FINISHERS = {"bag", "accessory"}


@dataclass
class Hit:
    product: dict
    score: float
    reasons: list[str]


@dataclass
class Look:
    hits: list[Hit]

    @property
    def total(self) -> float:
        return sum(h.product["price_eur"] for h in self.hits)


def passes(p: dict, q: Query, gender: str | None = None) -> bool:
    """True if the product meets every hard constraint (budget per item)."""
    g = gender or q.gender
    if g and p["gender"] not in (g, "unisex"):
        return False
    if q.categories and p["category"] not in q.categories:
        return False
    if q.season and p["season"] not in (q.season, "all-season"):
        return False
    if q.occasions and not set(q.occasions) & set(p["occasions"]):
        return False
    return not (q.budget is not None and (p["price_eur"] is None or p["price_eur"] > q.budget))


def reasons(p: dict, q: Query) -> list[str]:
    """Why this item: the constraints it matched, in plain words."""
    out = [o for o in q.occasions if o in p["occasions"]]
    if q.season:
        out.append(p["season"])
    if q.gender:
        out.append(p["gender"])
    if q.categories:
        out.append(p["category"])
    if q.budget is not None:
        out.append(f"€{p['price_eur']:g} ≤ €{q.budget:g}")
    return out


def rank(
    products: list[dict],
    vectors: np.ndarray,
    query_vec: np.ndarray,
    q: Query,
    gender: str | None = None,
) -> list[Hit]:
    """Products passing the hard filters, best semantic match first."""
    scores = vectors @ query_vec
    hits = [
        Hit(p, float(scores[i]), reasons(p, q))
        for i, p in enumerate(products)
        if passes(p, q, gender)
    ]
    return sorted(hits, key=lambda h: h.score, reverse=True)


def _best(hits: list[Hit], categories: set[str], n: int) -> list[Hit]:
    return [h for h in hits if h.product["category"] in categories][:n]


def complete_look(
    products: list[dict], vectors: np.ndarray, query_vec: np.ndarray, q: Query, n: int = 5
) -> Look | None:
    """Best outfit (dress, or top + bottom; then shoes and a bag or accessory) within the TOTAL budget.

    Tries the n best items per slot exhaustively (at most n^4 combinations) and keeps the highest
    mean semantic score. Without a gender in the query, each look stays within one gender (+ unisex).
    Returns None when no complete outfit fits.
    """
    best: tuple[float, Look] | None = None
    for gender in [q.gender] if q.gender else ["women", "men"]:
        hits = rank(products, vectors, query_vec, q, gender)
        shoes, finish = _best(hits, {"shoes"}, n), _best(hits, FINISHERS, n)
        bases = [(d,) for d in _best(hits, {"dress"}, n)]
        bases += list(cartesian(_best(hits, TOPS, n), _best(hits, BOTTOMS, n)))
        for base, s, f in cartesian(bases, shoes, finish):
            look = Look([*base, s, f])
            if q.budget is not None and look.total > q.budget:
                continue
            score = sum(h.score for h in look.hits) / len(look.hits)
            if best is None or score > best[0]:
                best = (score, look)
    if best is None:
        return None
    look = best[1]
    if q.budget is not None:  # in a look, the budget applies to the total, not to each item
        for h in look.hits:
            h.reasons = [r for r in h.reasons if not r.startswith("€")]
    return look
