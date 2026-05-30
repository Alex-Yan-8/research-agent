"""
Streamlit UI for the research agent.

Flow: input -> plan (editable) -> run (streaming) -> done (briefing + citations).

Run locally:  streamlit run app.py
Deploy via:   Streamlit Community Cloud (see README)
"""

import os
from io import BytesIO

import streamlit as st

try:
    for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from agent import format_citations, plan_research, run_research

CITATION_STYLES = [
    "APA (7th)",
    "MLA (9th)",
    "Chicago (author-date)",
    "Chicago (notes-bibliography)",
    "IEEE",
    "Harvard",
    "BibTeX",
]

st.set_page_config(page_title="Research Agent", layout="centered")
st.title("Research Agent")
st.caption("Plan, edit, and run web + private-corpus research. Adversarial mode and citation styles available.")


def parse_uploads(uploaded_files) -> dict:
    corpus = {}
    for f in uploaded_files or []:
        name = f.name
        try:
            if name.lower().endswith(".pdf"):
                import pypdf
                reader = pypdf.PdfReader(BytesIO(f.read()))
                text = "\n".join((p.extract_text() or "") for p in reader.pages)
            else:
                text = f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            text = f"[error parsing {name}: {e}]"
        corpus[name] = text
    return corpus


def reset() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


if "phase" not in st.session_state:
    st.session_state.phase = "input"


# ----- Phase: input -----------------------------------------------------------

if st.session_state.phase == "input":
    with st.form("input"):
        question = st.text_area(
            "Your question",
            height=100,
            placeholder="e.g. What's the current state of small-language-model fine-tuning?",
        )
        col1, col2 = st.columns(2)
        with col1:
            adversarial = st.checkbox(
                "Adversarial mode",
                value=False,
                help="Find counterevidence and stress-test the question.",
            )
        with col2:
            citation_style = st.selectbox("Citation style", CITATION_STYLES, index=0)

        uploaded = st.file_uploader(
            "Upload PDFs or text files (optional - the agent will search them alongside the web)",
            accept_multiple_files=True,
            type=["pdf", "txt", "md"],
        )
        submitted = st.form_submit_button("Generate plan", type="primary")

    if submitted and question.strip():
        st.session_state.question = question.strip()
        st.session_state.corpus = parse_uploads(uploaded)
        st.session_state.adversarial = adversarial
        st.session_state.citation_style = citation_style
        with st.spinner("Generating research plan..."):
            st.session_state.plan = plan_research(
                question.strip(),
                adversarial=adversarial,
                has_corpus=bool(st.session_state.corpus),
            )
        st.session_state.phase = "plan"
        st.rerun()


# ----- Phase: plan ------------------------------------------------------------

elif st.session_state.phase == "plan":
    st.subheader("Research plan")
    st.caption("Edit anything you want. The agent will use this as a guide.")

    badges = []
    if st.session_state.adversarial:
        badges.append("Adversarial mode ON")
    if st.session_state.corpus:
        badges.append(f"Corpus: {', '.join(st.session_state.corpus.keys())}")
    badges.append(f"Citation style: {st.session_state.citation_style}")
    st.info(" | ".join(badges))

    edited_plan = st.text_area(
        "Plan (editable)",
        value=st.session_state.plan,
        height=240,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("Approve and run", type="primary"):
            st.session_state.plan = edited_plan
            st.session_state.phase = "run"
            st.rerun()
    with col2:
        if st.button("Regenerate"):
            with st.spinner("Regenerating..."):
                st.session_state.plan = plan_research(
                    st.session_state.question,
                    adversarial=st.session_state.adversarial,
                    has_corpus=bool(st.session_state.corpus),
                )
            st.rerun()
    with col3:
        if st.button("Start over"):
            reset()
            st.rerun()


# ----- Phase: run -------------------------------------------------------------

elif st.session_state.phase == "run":
    st.subheader("Agent activity")
    activity_placeholder = st.empty()
    activity_log: list[str] = []

    def push_activity(line: str) -> None:
        activity_log.append(line)
        activity_placeholder.markdown("\n\n".join(activity_log))

    push_activity("_Starting..._")

    st.subheader("Briefing")
    briefing_placeholder = st.empty()
    text_chunks: list[str] = []

    def on_event(ev: dict) -> None:
        t = ev["type"]
        if t == "text_delta":
            text_chunks.append(ev["text"])
            briefing_placeholder.markdown("".join(text_chunks))
        elif t == "search_query":
            push_activity(f"**Search:** `{ev['query']}`")
        elif t == "search_results":
            urls = ev["urls"]
            shown = "\n".join(f"  - {u}" for u in urls[:8])
            extra = f"\n  - _...and {len(urls) - 8} more_" if len(urls) > 8 else ""
            push_activity(f"**Results:**\n{shown}{extra}")
        elif t == "fetch":
            push_activity(f"**Read:** {ev['url']}")
        elif t == "corpus_search":
            push_activity(f"**Corpus search:** `{ev['query']}`")

    with st.spinner("Researching..."):
        result = run_research(
            st.session_state.question,
            plan=st.session_state.plan,
            adversarial=st.session_state.adversarial,
            corpus=st.session_state.corpus or None,
            on_event=on_event,
        )

    if result["sources"]:
        with st.spinner(f"Formatting citations in {st.session_state.citation_style}..."):
            result["citations"] = format_citations(result["sources"], st.session_state.citation_style)
    else:
        result["citations"] = ""

    st.session_state.result = result
    st.session_state.activity_log = activity_log
    st.session_state.phase = "done"
    st.rerun()


# ----- Phase: done ------------------------------------------------------------

elif st.session_state.phase == "done":
    result = st.session_state.result

    with st.expander("Agent activity", expanded=False):
        st.markdown("\n\n".join(st.session_state.activity_log))

    st.subheader("Briefing")
    st.markdown(result["briefing"])

    if result["citations"]:
        st.subheader(f"Sources - {st.session_state.citation_style}")
        st.markdown(result["citations"])

        with st.expander("Raw URLs", expanded=False):
            for i, url in enumerate(result["sources"], 1):
                st.markdown(f"{i}. {url}")

    st.divider()
    if st.button("Start over", type="primary"):
        reset()
        st.rerun()
