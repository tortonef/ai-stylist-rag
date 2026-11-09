"""Deterministic query parser: budget, gender, category, season and occasion from free text.

Rules, not a model: the hard filters must be predictable and testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_NUM = r"(\d+(?:[.,]\d{1,2})?)"
_CUR = r"(?:€|eur\b|euros?\b)"
# "under €150", "less than 200 euros", "max 90", "budget of €300", "within 250"
_BUDGET_BEFORE = re.compile(
    r"\b(?:under|below|less than|up to|max(?:imum)?|within|no more than|at most|budget(?: of)?|for)\s*"
    rf"{_CUR}?\s*{_NUM}\s*({_CUR})?",
    re.IGNORECASE,
)
# "€300 max", "300 euros or less", "€200 budget"
_BUDGET_AFTER = re.compile(
    rf"(?:{_CUR}\s*{_NUM}|{_NUM}\s*{_CUR})\s*(?:or less|max(?:imum)?|budget|tops)\b", re.IGNORECASE
)
_MAY = re.compile(r"\b(?:in|early|late|mid|this|next)[\s-]+may\b", re.IGNORECASE)


@dataclass
class Query:
    """Hard constraints parsed from a question. Empty lists / None mean 'no constraint'."""

    text: str
    budget: float | None = None
    gender: str | None = None
    categories: list[str] = field(default_factory=list)
    season: str | None = None
    occasions: list[str] = field(default_factory=list)

    @property
    def wants_look(self) -> bool:
        """An occasion without a specific category asks for a complete outfit."""
        return bool(self.occasions) and not self.categories


def load_vocabulary(path: Path) -> dict[str, dict[str, list[str]]]:
    """Read config/vocabulary.yaml."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _matches(text: str, groups: dict[str, list[str]]) -> list[str]:
    """Schema values whose phrases occur in text, in order of first appearance, longest phrase first."""
    phrase_to_value = {ph.lower(): value for value, phrases in groups.items() for ph in phrases}
    alternation = "|".join(re.escape(ph) for ph in sorted(phrase_to_value, key=len, reverse=True))
    found: list[str] = []
    for m in re.finditer(rf"(?<![\w-])(?:{alternation})(?![\w-])", text, re.IGNORECASE):
        value = phrase_to_value[m.group(0).lower()]
        if value not in found:
            found.append(value)
    return found


def _budget(text: str) -> float | None:
    """Smallest amount stated as an upper limit, or None. A bare 'for 2' (no currency) is ignored."""
    amounts = []
    for m in _BUDGET_BEFORE.finditer(text):
        is_for = m.group(0).lower().startswith("for")
        has_currency = "€" in m.group(0) or re.search(_CUR, m.group(0), re.IGNORECASE)
        if is_for and not has_currency:
            continue
        amounts.append(m.group(1))
    for m in _BUDGET_AFTER.finditer(text):
        amounts.append(m.group(1) or m.group(2))
    values = [float(a.replace(",", ".")) for a in amounts]
    return min(values) if values else None


def parse_query(text: str, vocab: dict) -> Query:
    """Extract the hard constraints from a shopper's question."""
    genders = _matches(text, vocab["genders"])
    seasons = _matches(text, vocab["seasons"])
    if _MAY.search(text):
        seasons.append("SS")
    return Query(
        text=text,
        budget=_budget(text),
        gender=genders[0] if len(genders) == 1 else None,  # "for him and her" -> no gender filter
        categories=_matches(text, vocab["categories"]),
        season=seasons[0] if len(set(seasons)) == 1 else None,
        occasions=_matches(text, vocab["occasions"]),
    )
