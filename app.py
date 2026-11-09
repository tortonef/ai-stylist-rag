"""Streamlit UI: chat with the stylist; every recommendation is a catalog product card."""

from __future__ import annotations

import base64
import html

import streamlit as st

from ai_stylist.catalog import load_catalog
from ai_stylist.config import load_config
from ai_stylist.embed import load_encoder
from ai_stylist.index import load_or_build
from ai_stylist.llm import unavailable_reason
from ai_stylist.query import load_vocabulary
from ai_stylist.stylist import Answer, answer

st.set_page_config(page_title="AI Stylist", layout="wide")

EXAMPLES = [
    "What should I wear to a country wedding in May, under €300?",
    "A men's outfit for the office, under €450",
    "Something for a night out, she likes black",
    "What to pack for a summer holiday? Women's, under €250",
    "Running shoes and a gym bag under €200",
    "A warm jacket for a winter trip",
]
OCCASIONS = ["casual", "work", "evening", "ceremony", "sport", "travel"]
CATEGORIES = [
    "top", "shirt", "knitwear", "dress", "skirt", "trousers", "jeans",
    "outerwear", "shoes", "bag", "accessory",
]  # fmt: skip
AUTO = "Auto (from question)"


@st.cache_resource(show_spinner="Loading the catalog and the search index…")
def load_all():
    cfg = load_config()
    p = cfg["paths"]
    products = load_catalog(p["catalog"], p["schema"])
    encoder = load_encoder(cfg)
    vectors = load_or_build(cfg, products, encoder)
    return cfg, products, encoder, vectors, load_vocabulary(p["vocabulary"])


@st.cache_data(ttl=20, show_spinner=False)
def ollama_problem() -> str | None:
    return unavailable_reason(load_all()[0])


@st.cache_data
def thumbnail_uri(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")


try:
    cfg, products, encoder, vectors, vocab = load_all()
except RuntimeError as e:
    st.error(f"{e}\n\nRun `setup.ps1` (Windows) or `bash setup.sh` (macOS/Linux) first.")
    st.stop()
history: list[Answer] = st.session_state.setdefault("history", [])


# --- sidebar --------------------------------------------------------------------------------------

problem = ollama_problem()
with st.sidebar:
    st.subheader("Mode")
    use_llm = problem is None and st.toggle(
        "Write answers with the local LLM",
        value=True,
        help="Off = retrieval-only: instant results with a template explanation.",
    )
    if use_llm:
        st.success(f"**Full** · {cfg['ollama']['model']} via Ollama")
    else:
        st.warning("**Retrieval-only** · no generated text")

    st.subheader("Filters")
    st.caption("Parsed from your question automatically; set one here to override it.")
    gender = st.selectbox("Gender", [AUTO, "women", "men"])
    budget = st.number_input("Max budget (€, 0 = auto)", min_value=0, step=10)
    occasions = st.multiselect("Occasion", OCCASIONS)
    season = st.selectbox("Season", [AUTO, "SS", "AW"])
    categories = st.multiselect("Category", CATEGORIES)
    overrides = {
        "gender": None if gender == AUTO else gender,
        "budget": float(budget) or None,
        "occasions": occasions,
        "season": None if season == AUTO else season,
        "categories": categories,
    }

    if history and history[-1].look:
        st.subheader("Outfit total")
        st.metric("Last look", f"€{history[-1].look.total:g}")
    if history and st.button("Clear chat"):
        history.clear()
        st.rerun()


# --- main -----------------------------------------------------------------------------------------

st.title("AI Stylist")
st.caption(
    "Ask for an outfit or a product. Answers use only products from the Atelier Nove catalog "
    "(fictional), each one traceable to a SKU. Everything runs on this machine."
)
if problem:
    st.info(
        f"**Retrieval-only mode.** {problem} Results come from the catalog search with a template "
        "explanation. For generated answers: install Ollama from https://ollama.com/download, run "
        f"`ollama pull {cfg['ollama']['model']}`, then reload this page."
    )


def card(hit) -> str:
    p = hit.product
    chips = "".join(
        f'<span style="background:#E3ECE4;color:#1F1B24;border-radius:10px;padding:1px 8px;'
        f'margin:2px 4px 0 0;font-size:0.72rem;display:inline-block">{html.escape(r)}</span>'
        for r in hit.reasons
    )
    return (
        '<div style="background:#FFFFFF;border:1px solid #E2D8C9;border-radius:10px;padding:10px;height:100%">'
        f'<img src="{thumbnail_uri(str(cfg["paths"]["thumbnails"] / (p["sku"] + ".svg")))}" '
        f'alt="{html.escape(p["name"])}" style="width:100%;aspect-ratio:200/172;object-fit:cover;'
        'object-position:top;border-radius:8px"/>'
        f'<div style="font-weight:600;margin-top:6px">{html.escape(p["name"])}</div>'
        f'<div style="font-size:0.8rem;opacity:0.7">{p["sku"]} · {html.escape(", ".join(p["colors"]))}</div>'
        f'<div style="font-size:1.05rem;font-weight:600;margin:4px 0">€{p["price_eur"]:g}</div>'
        f'<div title="Why this item: the constraints it matched">{chips}</div></div>'
    )


def render(a: Answer) -> None:
    with st.chat_message("user"):
        st.write(a.question)
    with st.chat_message("assistant"):
        if a.note:
            st.info(a.note)
        if a.text and a.hits:
            st.markdown(a.text)
        if a.removed:
            st.warning(
                "Grounding check removed SKU(s) the model mentioned that were not retrieved for this "
                f"question: {', '.join(a.removed)}."
            )
        if a.error:
            st.warning(f"The LLM failed, showing the retrieval-only answer instead. ({a.error})")
        if a.hits:
            cols = st.columns(min(len(a.hits), 4))
            for i, h in enumerate(a.hits):
                cols[i % len(cols)].markdown(card(h), unsafe_allow_html=True)
        if a.look:
            budget = f" (budget €{a.query.budget:g})" if a.query.budget is not None else ""
            st.markdown(f"**Outfit total: €{a.look.total:g}**{budget} · {len(a.look.hits)} pieces")
        timing = f"retrieval {a.retrieval_s * 1000:.0f} ms"
        if a.mode == "full":
            timing += f" · generation {a.generation_s:.1f} s"
        st.caption(f"{a.mode} mode · {timing}")


if not history:
    st.markdown("**Try an example:**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, use_container_width=True):
            st.session_state["pending"] = ex

for a in history:
    render(a)

question = st.chat_input("e.g. What should I wear to a country wedding in May, under €300?")
question = question or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.write(question)
    spinner = "Writing the answer with the local LLM…" if use_llm else "Searching the catalog…"
    with st.spinner(spinner):
        a = answer(question, cfg, products, vectors, encoder, vocab, use_llm, overrides)
    history.append(a)
    st.rerun()
