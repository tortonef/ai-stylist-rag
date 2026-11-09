from itertools import combinations

from ai_stylist.query import Query, parse_query
from ai_stylist.search import BOTTOMS, TOPS, complete_look, passes, rank


def _vec(encoder, text):
    return encoder.encode([text])[0]


def test_hard_filters_are_respected(products, vectors, encoder, vocab):
    q = parse_query("women's shoes for a wedding under €120 in summer", vocab)
    hits = rank(products, vectors, _vec(encoder, q.text), q)
    assert hits
    for h in hits:
        p = h.product
        assert p["category"] == "shoes" and p["price_eur"] <= 120
        assert p["gender"] in ("women", "unisex") and p["season"] in ("SS", "all-season")
        assert "ceremony" in p["occasions"]


def test_every_passing_product_is_returned(products, vectors, encoder):
    q = Query(text="x", gender="men", occasions=["work"])
    hits = rank(products, vectors, _vec(encoder, "x"), q)
    assert {h.product["sku"] for h in hits} == {p["sku"] for p in products if passes(p, q)}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_reasons_explain_the_match(products, vectors, encoder):
    q = Query(text="x", budget=100, occasions=["ceremony"], categories=["shoes"])
    hit = rank(products, vectors, _vec(encoder, "x"), q)[0]
    assert "ceremony" in hit.reasons and "shoes" in hit.reasons
    assert any(r.startswith("€") for r in hit.reasons)


def _check_look(look, q):
    cats = [h.product["category"] for h in look.hits]
    assert cats[-2] == "shoes" and cats[-1] in ("bag", "accessory")
    assert cats[:-2] == ["dress"] or (cats[0] in TOPS and cats[1] in BOTTOMS)
    genders = {h.product["gender"] for h in look.hits} - {"unisex"}
    assert len(genders) <= 1, "a look never mixes women's and men's items"
    for h in look.hits:
        assert set(q.occasions) & set(h.product["occasions"])
        assert not any(r.startswith("€") for r in h.reasons)  # budget is on the total


def test_look_stays_within_total_budget(products, vectors, encoder, vocab):
    q = parse_query("What should I wear to a country wedding in May, under €300?", vocab)
    look = complete_look(products, vectors, _vec(encoder, q.text), q)
    assert look is not None and look.total <= 300
    _check_look(look, q)


def test_look_is_the_best_affordable_combination(products, vectors, encoder):
    # With a single-gender, small candidate pool, compare against a brute force over everything.
    q = Query(text="office", gender="men", occasions=["work"], budget=420)
    qv = _vec(encoder, q.text)
    look = complete_look(products, vectors, qv, q, n=50)
    hits = {h.product["sku"]: h for h in rank(products, vectors, qv, q)}
    best = max(
        (
            sum(hits[s].score for s in combo) / 4
            for combo in combinations(hits, 4)
            if sum(hits[s].product["price_eur"] for s in combo) <= 420
            and _is_separates_look([hits[s].product["category"] for s in combo])
        ),
    )
    assert abs(sum(h.score for h in look.hits) / len(look.hits) - best) < 1e-6
    _check_look(look, q)


def _is_separates_look(cats):
    return (
        sum(c in TOPS for c in cats) == 1
        and sum(c in BOTTOMS for c in cats) == 1
        and cats.count("shoes") == 1
        and sum(c in ("bag", "accessory") for c in cats) == 1
    )


def test_no_look_when_budget_too_low(products, vectors, encoder):
    q = Query(text="wedding", gender="men", occasions=["ceremony"], budget=200)
    assert complete_look(products, vectors, _vec(encoder, q.text), q) is None
