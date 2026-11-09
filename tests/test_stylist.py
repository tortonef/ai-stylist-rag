import json

import pytest

from ai_stylist import llm
from ai_stylist.stylist import SKU, answer, build_messages, ground


def test_ground_keeps_allowed_and_removes_unknown():
    text = "Try the dress [AN-0001] with [AN-0999] sandals [AN-0029, AN-0500] and AN-0777."
    clean, cited, removed = ground(text, {"AN-0001", "AN-0029"})
    assert clean == "Try the dress [AN-0001] with sandals [AN-0029] and."
    assert cited == ["AN-0001", "AN-0029"]
    assert removed == ["AN-0999", "AN-0500", "AN-0777"]


def test_ground_leaves_plain_text_alone():
    assert ground("No products here [see below].", {"AN-0001"}) == (
        "No products here [see below].",
        [],
        [],
    )


def test_catalog_sku_not_retrieved_is_still_removed():
    # AN-0002 exists in the catalog, but was not retrieved for this question.
    _, cited, removed = ground("[AN-0001] or [AN-0002]", {"AN-0001"})
    assert cited == ["AN-0001"] and removed == ["AN-0002"]


def _run(cfg, products, vectors, encoder, vocab, question, chat=None, use_llm=True, overrides=None):
    return answer(
        question,
        cfg,
        products,
        vectors,
        encoder,
        vocab,
        use_llm,
        overrides,
        chat or (lambda m, c: ""),
    )


def test_full_mode_grounds_the_model_answer(cfg, products, vectors, encoder, vocab):
    seen = {}

    def fake_chat(messages, _cfg):
        seen["prompt"] = messages[-1]["content"]
        first = SKU.search(seen["prompt"]).group(0)  # a retrieved product
        return f"Wear [{first}] and the invented [ZZ-1234]."

    a = _run(cfg, products, vectors, encoder, vocab, "a wedding in May under €300", fake_chat)
    assert a.mode == "full" and a.look is not None
    assert a.removed == ["ZZ-1234"] and "ZZ-1234" not in a.text
    assert len(a.cited) == 1 and set(a.cited) <= {h.product["sku"] for h in a.hits}
    # The prompt holds only the retrieved products.
    in_prompt = {p["sku"] for p in products if f"[{p['sku']}]" in seen["prompt"]}
    assert in_prompt == {h.product["sku"] for h in a.hits}


def test_llm_failure_falls_back_to_template(cfg, products, vectors, encoder, vocab):
    def broken(messages, _cfg):
        raise llm.LLMError("Cannot reach Ollama")

    a = _run(cfg, products, vectors, encoder, vocab, "linen shirt for him", broken)
    assert a.mode == "retrieval-only" and a.error and a.hits
    assert all(f"[{h.product['sku']}]" in a.text for h in a.hits)


def test_retrieval_only_mode_never_calls_the_llm(cfg, products, vectors, encoder, vocab):
    def must_not_run(messages, _cfg):
        raise AssertionError("LLM called in retrieval-only mode")

    a = _run(
        cfg,
        products,
        vectors,
        encoder,
        vocab,
        "office look for her under €400",
        must_not_run,
        use_llm=False,
    )
    assert a.mode == "retrieval-only" and a.look is not None
    assert f"Total €{a.look.total:g}" in a.text and a.look.total <= 400


def test_no_match_gives_a_clear_note(cfg, products, vectors, encoder, vocab):
    a = _run(cfg, products, vectors, encoder, vocab, "a coat under €10", use_llm=False)
    assert a.hits == [] and "Nothing in the catalog" in a.text


def test_impossible_look_falls_back_to_items(cfg, products, vectors, encoder, vocab):
    a = _run(
        cfg, products, vectors, encoder, vocab, "men's wedding outfit under €250", use_llm=False
    )
    assert a.look is None and a.hits and "No complete outfit" in a.note
    assert all(h.product["price_eur"] <= 250 for h in a.hits)


def test_sidebar_overrides_replace_parsed_filters(cfg, products, vectors, encoder, vocab):
    a = _run(
        cfg,
        products,
        vectors,
        encoder,
        vocab,
        "a linen shirt",
        use_llm=False,
        overrides={"gender": "women"},
    )
    assert a.query.gender == "women"
    assert all(h.product["gender"] in ("women", "unisex") for h in a.hits)


def test_build_messages_mentions_the_rules():
    msgs = build_messages("q", [], None)
    assert "square brackets" in msgs[0]["content"] and "Do not state prices" in msgs[0]["content"]


@pytest.mark.parametrize(
    "content, ok",
    [('{"answer": " Hi [AN-0001] "}', "Hi [AN-0001]"), ("not json", None), ('{"text": "x"}', None)],
)
def test_parse_answer(content, ok):
    if ok:
        assert llm.parse_answer(content) == ok
    else:
        with pytest.raises(llm.LLMError):
            llm.parse_answer(content)


def test_chat_sends_think_false_and_schema(cfg, monkeypatch):
    captured = {}

    def fake_request(url, body, timeout):
        captured.update(url=url, body=body)
        return {"message": {"content": json.dumps({"answer": "ok"})}}

    monkeypatch.setattr(llm, "_request", fake_request)
    assert llm.chat([{"role": "user", "content": "hi"}], cfg) == "ok"
    assert captured["body"]["think"] is False and captured["body"]["format"] == llm.ANSWER_SCHEMA
    assert captured["url"].endswith("/api/chat")


def test_unavailable_reason(cfg, monkeypatch):
    monkeypatch.setattr(llm, "_request", lambda *a: (_ for _ in ()).throw(OSError("down")))
    assert "not running" in llm.unavailable_reason(cfg)
    monkeypatch.setattr(llm, "_request", lambda *a: {"models": [{"name": "other:1b"}]})
    assert "not pulled" in llm.unavailable_reason(cfg)
    monkeypatch.setattr(llm, "_request", lambda *a: {"models": [{"name": cfg["ollama"]["model"]}]})
    assert llm.unavailable_reason(cfg) is None
