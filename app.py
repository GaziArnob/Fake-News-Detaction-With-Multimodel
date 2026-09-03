"""Streamlit upload interface for the Fake News Detection project."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from news_guard.config import get_settings
from news_guard.service import NewsVerificationService


st.set_page_config(page_title="SignalCheck", page_icon="◉", layout="wide")

st.markdown(
    """
    <style>
      .stApp, [data-testid="stAppViewContainer"] { background: #0b1010; color: #edf5f1; }
      header, [data-testid="stHeader"] { background: #0b1010 !important; }
      [data-testid="stSidebar"] { background: #111918; border-right: 1px solid #27332f; }
      [data-testid="stSidebar"] * { color: #dce8e2; }
      .block-container { max-width: 1180px; padding-top: 2.4rem; }
      .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong,
      [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
      [data-testid="stMarkdownContainer"] li { color: #edf5f1 !important; }
      .eyebrow { color: #6ee7bb !important; font-weight: 700; letter-spacing: .11em; font-size: .78rem; text-transform: uppercase; }
      .hero-title { font-size: clamp(2.6rem, 6vw, 5.2rem); line-height: .92; letter-spacing: -.07em; margin: .45rem 0 1rem; color: #f4fbf7 !important; }
      .hero-copy { max-width: 680px; color: #c2d0c9 !important; font-size: 1.1rem; }
      .section-label { color: #6ee7bb !important; font-weight: 700; font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; margin-top: 2rem; }
      label, .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #bfd0c7 !important; }
      [data-testid="stMetric"] { background: #15211e; border: 1px solid #29443a; padding: .85rem 1rem; border-radius: .4rem; }
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: #f2faf6 !important; }
      .muted { color: #aebfb6 !important; }
      div[data-testid="stFileUploader"] { background: #15211e; border: 1px dashed #477664; padding: 1.25rem; border-radius: .55rem; }
      div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color: #d7e5de !important; }
      input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
      [data-baseweb="select"] > div { background: #101817 !important; color: #f1f8f4 !important; border-color: #426353 !important; }
      [data-baseweb="select"] *, [role="listbox"] *, [role="option"] { color: #f1f8f4 !important; }
      [role="listbox"], [role="option"] { background: #16221f !important; }
      .stButton > button { background: #1f8a6d; color: #ffffff !important; border: none; border-radius: .32rem; font-weight: 700; padding: .65rem 1rem; }
      .stButton > button:hover { background: #29a783; color: #ffffff !important; }
      [data-testid="stAlert"] { color: #f4faf7 !important; }
      [data-testid="stAlert"] * { color: inherit !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def service() -> NewsVerificationService:
    return NewsVerificationService(get_settings())


def save_upload(uploaded_file) -> Path:
    settings = get_settings()
    contents = uploaded_file.getvalue()
    digest = hashlib.sha256(contents).hexdigest()[:20]
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    path = settings.upload_path / f"{digest}{suffix}"
    path.write_bytes(contents)
    return path


def render_result(result: dict) -> None:
    local = result["local_prediction"]
    fake_percent = local["fake_probability"] * 100
    st.markdown("<div class='section-label'>Local multimodal assessment</div>", unsafe_allow_html=True)
    metric_col, detail_col = st.columns([1, 2], gap="large")
    with metric_col:
        st.metric("Fake-news risk", f"{fake_percent:.1f}%")
        if local["label"] == "fake":
            st.error("Dataset model result: likely fake")
        else:
            st.success("Dataset model result: likely real")
    with detail_col:
        st.write("**Extracted claim**")
        st.write(local["claim"] or "No readable claim was extracted; review the image manually.")
        st.caption(f"Caption: {local['caption'] or '—'}")

    st.markdown("<div class='section-label'>Dual-model verification</div>", unsafe_allow_html=True)
    verification = result["dual_verification"]
    st.write(f"**Consensus:** {verification['consensus'] or verification['status'].replace('_', ' ')}")
    gemini_col, groq_col = st.columns(2, gap="large")
    for column, title, verdict in (
        (gemini_col, "Gemini", verification["gemini"]),
        (groq_col, "Groq", verification["groq"]),
    ):
        with column:
            st.write(f"**{title} — {verdict.get('status', 'not run')}**")
            if verdict.get("verdict"):
                st.write(f"{verdict['verdict']} · {verdict.get('confidence', 0):.0f}%")
            st.caption(verdict.get("rationale", ""))

    st.markdown("<div class='section-label'>Retrieved evidence</div>", unsafe_allow_html=True)
    evidence = result.get("evidence", [])
    st.caption(
        f"Local RAG matches: {result.get('local_evidence_count', 0)} · "
        f"Live search matches: {result.get('live_evidence_count', 0)} "
        f"({result.get('live_search_status', 'not_configured')})"
    )
    if evidence:
        with st.expander(f"Show {len(evidence)} retrieved source(s) sent to Gemini/Groq", expanded=False):
            for item in evidence:
                score = item.get("score")
                score_text = f" · similarity {score:.2f}" if score is not None else ""
                st.markdown(f"**[{item['source_type']}] {item['title']}**{score_text}")
                if item.get("url"):
                    st.caption(item["url"])
                st.write(item.get("text", ""))
                st.divider()
    else:
        st.caption("No sources were retrieved for this claim — the verdict above was based on the claim text alone.")

    st.warning(result["disclaimer"])


st.markdown("<div class='eyebrow'>Image claim verification workspace</div>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>Signal<br>Check.</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-copy'>Inspect a news image, review the model assessment, and send human-approved corrections into a controlled update queue.</p>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a JPG, JPEG, or PNG news image", type=["jpg", "jpeg", "png"])
analyze = st.button("Analyze image", type="primary", disabled=uploaded is None)

if analyze and uploaded is not None:
    saved_path = save_upload(uploaded)
    st.image(saved_path, caption=uploaded.name, use_container_width=True)
    with st.spinner("Extracting image features…"):
        try:
            result = service().analyze(saved_path)
            st.session_state["last_result"] = result
            st.session_state["last_image_path"] = str(saved_path)
        except Exception as exc:
            st.error(str(exc))

if "last_result" in st.session_state:
    render_result(st.session_state["last_result"])
    st.markdown("<div class='section-label'>Human-approved update queue</div>", unsafe_allow_html=True)
    with st.form("human_verdict"):
        reviewer = st.text_input("Reviewer name or ID")
        approved_label = st.radio("Verified label", ["real", "fake"], horizontal=True)
        source_url = st.text_input("Evidence/source URL (optional)")
        notes = st.text_area("Review notes (optional)")
        save_verdict = st.form_submit_button("Save approved example for future retraining")
    if save_verdict:
        try:
            record = service().save_human_verdict(
                st.session_state["last_image_path"], approved_label, reviewer, source_url, notes
            )
            st.success(f"Saved approved example {record['record_id'][:8]} to the update queue.")
        except Exception as exc:
            st.error(str(exc))
