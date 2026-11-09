from ai_stylist.evaluate import parsed_ok, violations
from ai_stylist.query import Query

DRESS = {
    "gender": "women",
    "category": "dress",
    "season": "SS",
    "occasions": ["ceremony"],
    "price_eur": 149,
}


def test_violations():
    assert violations(DRESS, {"occasions": ["ceremony"], "budget": 200}, per_item_budget=True) == []
    assert violations(DRESS, {"gender": "men", "budget": 100}, per_item_budget=True) == [
        "gender",
        "budget",
    ]
    assert (
        violations(DRESS, {"budget": 100}, per_item_budget=False) == []
    )  # look: budget is on the total
    assert violations(DRESS, {"season": "AW", "categories": ["shoes"]}, True) == [
        "category",
        "season",
    ]


def test_parsed_ok():
    q = Query(text="x", budget=300, occasions=["ceremony"], season="SS")
    ok = parsed_ok(q, {"budget": 300, "occasions": ["ceremony"], "season": "SS", "look": True})
    assert all(ok.values())
    assert not parsed_ok(q, {"budget": 300, "occasions": ["work"], "season": "SS"})["occasions"]
