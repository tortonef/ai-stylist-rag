"""Real embedding model and real Ollama: run with `pytest -m slow` after setup, with Ollama running."""

import pytest

from ai_stylist import llm
from ai_stylist.embed import load_encoder
from ai_stylist.index import load_or_build
from ai_stylist.stylist import answer

pytestmark = pytest.mark.slow
QUESTION = "What should I wear to a country wedding in May, under €300?"


@pytest.fixture(scope="module")
def real(cfg, products):
    encoder = load_encoder(cfg)
    return encoder, load_or_build(cfg, products, encoder)


def test_real_retrieval_builds_a_look_within_budget(cfg, products, vocab, real):
    encoder, vectors = real
    a = answer(QUESTION, cfg, products, vectors, encoder, vocab, use_llm=False)
    assert a.look is not None and a.look.total <= 300
    assert any(h.product["category"] == "dress" for h in a.look.hits)


def test_real_llm_answer_is_grounded(cfg, products, vocab, real):
    if reason := llm.unavailable_reason(cfg):
        pytest.skip(reason)
    encoder, vectors = real
    a = answer(QUESTION, cfg, products, vectors, encoder, vocab, use_llm=True)
    assert a.mode == "full" and a.cited
    assert set(a.cited) <= {h.product["sku"] for h in a.hits}
