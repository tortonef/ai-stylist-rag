import pytest

from ai_stylist.query import parse_query


@pytest.mark.parametrize(
    "text, budget",
    [
        ("dress under €150", 150),
        ("shoes less than 200 euros", 200),
        ("a coat below 300 EUR", 300),
        ("bag, max 90", 90),
        ("budget of €250 for a wedding", 250),
        ("something for €120", 120),
        ("€300 max", 300),
        ("200 euros or less", 200),
        ("under 99.50 euro", 99.5),
        ("a dinner for 2", None),
        ("a blue linen shirt", None),
        ("under €300, ideally less than €200", 200),
    ],
)
def test_budget(vocab, text, budget):
    assert parse_query(text, vocab).budget == budget


def test_spec_example(vocab):
    q = parse_query("What should I wear to a country wedding in May, under €300?", vocab)
    assert (q.budget, q.season, q.occasions, q.categories, q.gender) == (
        300,
        "SS",
        ["ceremony"],
        [],
        None,
    )
    assert q.wants_look


@pytest.mark.parametrize(
    "text, occasions",
    [
        ("an outfit for the office", ["work"]),
        ("weekend in the city", ["casual"]),
        ("a night out with friends", ["evening"]),
        ("birthday party then dinner", ["evening"]),
        ("gym bag", ["sport"]),
        ("a holiday trip", ["travel"]),
        ("wedding guest", ["ceremony"]),
        ("a blue shirt", []),
    ],
)
def test_occasion_synonyms(vocab, text, occasions):
    assert parse_query(text, vocab).occasions == occasions


def test_gender_and_category(vocab):
    q = parse_query("men's linen shirt and loafers", vocab)
    assert q.gender == "men" and q.categories == ["shirt", "shoes"]
    assert not q.wants_look
    assert parse_query("gifts for him and her", vocab).gender is None
    assert parse_query("there is a theme", vocab).gender is None  # "he" inside words is ignored


def test_t_shirt_is_a_top_not_a_shirt(vocab):
    assert parse_query("white t-shirt", vocab).categories == ["top"]


def test_may_only_as_a_month(vocab):
    assert parse_query("a wedding in May", vocab).season == "SS"
    assert parse_query("what may I wear in winter", vocab).season == "AW"
    assert parse_query("summer or winter", vocab).season is None
