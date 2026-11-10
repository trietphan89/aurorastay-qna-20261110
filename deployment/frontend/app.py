"""AuroraStay staff policy lookup UI. No credentials are stored in the browser."""
import os
from html import escape
from pathlib import PurePath

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:7860").rstrip("/")
HTTP_TIMEOUT = (5, 95)
MODES = ["Answer only", "Answer + Relevant Chunks"]

st.set_page_config(page_title="AuroraStay | Policy desk", page_icon="🏨", layout="wide")
st.markdown("""
<style>
  .stApp { background: #fcfcfa; }
  .block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 3rem; }
  h1, h2, h3 { letter-spacing: -.025em; }
  [data-testid="stSidebar"] { background: #f2f3ef; }
  [data-testid="stSidebar"] h2 { font-size: 1.2rem; }
  [data-testid="stTextArea"] textarea { font-size: 1.05rem; line-height: 1.6; }
  .eyebrow { font-size: .75rem; font-weight: 700; letter-spacing: .16em; color: #34564c; margin-bottom: .5rem; }
  .review-note { border-left: 3px solid #9b7333; padding: .4rem 1rem; color: #5c492c; margin: 1rem 0 1.5rem; }
  .source-note { color: #66716a; font-size: .9rem; }
  .answer-copy { white-space: normal; overflow-wrap: anywhere; font-size: 1.05rem; line-height: 1.75; }
  .passage-copy { white-space: normal; overflow-wrap: anywhere; font-size: .9rem; line-height: 1.65; }
  .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] { background: #244e43; border-color: #244e43; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### AuroraStay")
    st.caption("INTERNAL POLICY DESK")
    st.divider()
    st.subheader("Lookup settings")
    k = st.slider("Policy passages", 1, 10, 4,
                  help="How many matching passages to use. Increase for questions spanning several policies.")
    temperature = st.slider("Response variability", 0.0, 1.0, 0.0, 0.1,
                            help="Keep this low for consistent policy wording.")
    max_tokens = st.slider("Response length limit", 64, 2048, 512, 64,
                           help="Maximum generated tokens. A value of 512 is a useful starting point.")
    st.divider()
    st.caption("Questions cover booking, property information, guest conduct, refunds and additional charges.")
    st.caption("Answers are drafts for staff review. Confirm property-specific conditions before sharing.")

st.markdown('<div class="eyebrow">AURORASTAY · STAFF ASSISTANT</div>', unsafe_allow_html=True)
st.title("Hotel policy lookup")
st.write("Ask about hotel policies and check the supporting documents before responding.")
st.divider()

with st.form("policy_question", clear_on_submit=False):
    response_mode = st.radio("Response mode", MODES, horizontal=True, index=1)
    query = st.text_area("Guest question", height=120, max_chars=2000,
                         placeholder="For ASH-SEA Pike, is breakfast included and what should I budget for parking?")
    submitted = st.form_submit_button("Find policy answer", type="primary")

if submitted:
    if not query.strip():
        st.session_state.pop("policy_result", None)
        st.warning("Enter a guest question to begin.")
    else:
        endpoint = "/v1/answer" if response_mode == MODES[0] else "/v1/answer_with_relevant_chunks"
        payload = {"query": query.strip(), "k": k, "model_name": "gpt-4o-mini",
                   "temperature": temperature, "top_p": 1.0, "max_tokens": max_tokens}
        try:
            with st.spinner("Finding the relevant hotel policies…"):
                response = requests.post(BACKEND_URL + endpoint, json=payload, timeout=HTTP_TIMEOUT)
            try:
                result = response.json()
            except ValueError:
                result = {}
            if response.ok and isinstance(result.get("answer"), str):
                st.session_state["policy_result"] = {"query": query.strip(), "mode": response_mode, "result": result}
            else:
                st.session_state.pop("policy_result", None)
                message = result.get("error", {}).get("message") if isinstance(result.get("error"), dict) else None
                st.error(message or "The assistant could not provide an answer. Please use manual lookup or try again.")
        except requests.Timeout:
            st.session_state.pop("policy_result", None)
            st.error("This lookup took too long. Please use manual lookup or try again.")
        except requests.RequestException:
            st.session_state.pop("policy_result", None)
            st.error("The policy assistant is currently unavailable. Please use the hotel policy documents directly.")

saved = st.session_state.get("policy_result")
if saved:
    result = saved["result"]
    st.divider()
    st.caption("QUESTION REVIEWED")
    st.write(saved["query"])
    st.subheader("Suggested answer")
    st.markdown('<div class="review-note">Staff review required · Verify the policy evidence before sharing with a guest.</div>', unsafe_allow_html=True)
    # Escape model output before inserting it into a wrapping text region.
    st.markdown('<div class="answer-copy">' + escape(result["answer"]).replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
    sources = result.get("sources", [])
    if sources:
        st.subheader("Supporting documents")
        seen = set()
        for source in sources:
            filename = PurePath(str(source.get("source") or source.get("document") or source.get("filename") or "Policy document")).name
            pages = source.get("page_numbers") or source.get("pages") or source.get("page")
            if isinstance(pages, list):
                pages = ", ".join(map(str, pages))
            chunk = source.get("chunk_id")
            label = filename + (f" · page {pages}" if pages is not None else "") + (f" · {chunk}" if chunk is not None else "")
            if label not in seen:
                st.caption(label)
                seen.add(label)
    if saved["mode"] == MODES[1]:
        st.subheader("Relevant policy passages")
        for index, passage in enumerate(result.get("relevant_chunks", []), 1):
            with st.expander(f"Passage {index}", expanded=index == 1):
                st.markdown('<div class="passage-copy">' + escape(str(passage)).replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
    st.caption("This lookup creates a draft only. It does not change a reservation, issue a refund or contact a guest.")
else:
    st.caption("Include a property code, rate type and relevant dates when available. If a policy is missing or unclear, use manual lookup or ask a supervisor.")
