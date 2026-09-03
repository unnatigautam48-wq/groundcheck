import streamlit as st

from evaluate import CORPUS
from pipeline import RAGPipeline

st.set_page_config(page_title="GroundCheck", page_icon="🔍", layout="centered")

st.title("🔍 GroundCheck")
st.caption(
    "A RAG pipeline that checks its own answers. Ask a question about "
    "NimbusCloud (a fictional SaaS company's support docs) and see whether "
    "the answer is actually backed by the source text."
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline(CORPUS)


pipeline = load_pipeline()

with st.expander("What's in the knowledge base?"):
    for chunk in CORPUS:
        st.markdown(f"**{chunk.doc_id}** — {chunk.text}")

query = st.text_input(
    "Ask a question",
    placeholder="e.g. How much does the Pro plan cost?",
)

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with col2:
    hallucinate_clicked = st.button(
        "Ask + inject a fake fact", use_container_width=True,
        help="Feeds in a deliberately wrong answer so you can see the "
             "detector catch it.",
    )

if ask_clicked and query:
    result = pipeline.ask(query)
    st.subheader("Answer")
    st.write(result.answer)
    st.metric("Groundedness score", result.score)
    for check in result.claim_checks:
        icon = {"SUPPORTED": "🟢", "WEAK": "🟡", "UNSUPPORTED": "🔴"}[check.label]
        with st.container(border=True):
            st.markdown(f"{icon} **{check.label}** — {check.claim}")
            if check.label != "SUPPORTED":
                reason = check.reason or f"closest match only {check.best_match_score} similar"
                st.caption(f"Flagged: {reason}")
                st.caption(f"Closest source line: \"{check.best_match_source}\"")

elif hallucinate_clicked and query:
    fake_answer = st.text_input(
        "Type a wrong answer to inject (swap a number or fact from the real answer)",
        key="fake_answer_input",
    )
    if fake_answer:
        result = pipeline.ask(query, injected_answer=fake_answer)
        st.subheader("Verification of your injected answer")
        st.metric("Groundedness score", result.score)
        for check in result.claim_checks:
            icon = {"SUPPORTED": "🟢", "WEAK": "🟡", "UNSUPPORTED": "🔴"}[check.label]
            with st.container(border=True):
                st.markdown(f"{icon} **{check.label}** — {check.claim}")
                if check.label != "SUPPORTED":
                    reason = check.reason or f"closest match only {check.best_match_score} similar"
                    st.caption(f"Flagged: {reason}")

st.divider()
st.caption(
    "Built with TF-IDF retrieval + a lexical/numeric/negation groundedness "
    "checker — no external LLM or API key required. "
    "[View source](https://github.com/unnatigautam48-wq/groundcheck)"
)
