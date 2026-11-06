# app.py
# streamlit run app.py

import streamlit as st

import config
from querying import display
from querying.pipeline import Options, answer_question, find_chunks

st.set_page_config(page_title="Onboarding Assistant")

SAMPLE_QUESTIONS = [
    "How many days of annual leave do I get?",
    "can i get paid back for my train ticket",
    "What is the price of item ITM-20417?",
    "What does VPN error 612 mean?",
    "What time does the cafeteria close?",
]
HAS_COMPANY_TIER = bool(config.COMPANY_SOURCE_PREFIXES)


@st.cache_resource(show_spinner="Loading the search models. This happens once after the app starts.")
def warm_up():
    """Load the embedding model, the keyword index, the vector database and the reranker."""
    find_chunks("warm up", Options(use_hybrid=True, use_rerank=True, company_only=HAS_COMPANY_TIER))
    return True


def render_assistant(message, show_details):
    st.markdown(display.escape_for_markdown(message["content"]))

    if message.get("warning"):
        st.warning("The best matching passage was only a loose match for this question. "
                   "Please check the sources below before relying on this answer.")

    cards = message.get("sources", [])
    if cards:
        label = "Sources used" if message["status"] == "answered" else "Passages that were checked"
        with st.expander(label):
            for card in cards:
                st.markdown(f"**[{card['number']}]** `{card['title']}`")
                st.caption(f"{card['section']}  |  match score {card['score']:.2f}")
                st.text(card["snippet"])

    if show_details:
        changed = message["search_query"] != message["question"]
        st.caption(f"Question searched  {message['search_query']}" + ("" if changed else "  (unchanged)"))
        timings = "  ".join(f"{name} {seconds}s" for name, seconds in message["timings"].items())
        score = "none" if message["top_score"] is None else f"{message['top_score']:.2f}"
        st.caption(f"Status {message['status']}  |  best score {score}  |  tokens {message['tokens']}  |  {timings}")


# ---------------------------------------------------------------------------
# Start up
# ---------------------------------------------------------------------------
try:
    warm_up()
except Exception as error:      # most often the knowledge base has not been built yet
    st.error("The knowledge base could not be opened. Run  python -m indexing.build_index  "
             f"in the project folder and start the app again.  Details  {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Onboarding Assistant")
st.caption("Answers come only from the documents in the knowledge base. "
           "Check the sources before you act on an answer.")

with st.sidebar:
    st.header("Settings")
    company_only = st.checkbox(
        "Search company documents only",
        value=HAS_COMPANY_TIER,
        disabled=not HAS_COMPANY_TIER,
        help="Leave this on for normal use. Switch it off to also search the public handbooks "
             "of other organizations, which is the harder test setting.",
    )
    rewrite_on = st.checkbox(
        "Rewrite questions before searching",
        value=True,
        help="Turns casual wording into the words the documents use. Costs one small extra model call.",
    )
    show_details = st.checkbox("Show how each answer was found", value=False)

    st.header("Try an example")
    for sample in SAMPLE_QUESTIONS:
        if st.button(sample):
            st.session_state.pending_question = sample

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

options = Options(use_hybrid=True, use_rerank=True, use_rewrite=rewrite_on, company_only=company_only)

# ---------------------------------------------------------------------------
# The conversation so far
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(display.escape_for_markdown(message["content"]))
        else:
            render_assistant(message, show_details)

# ---------------------------------------------------------------------------
# A new question, typed in the box or chosen from the examples
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about onboarding, IT, expenses, leave or warehouse safety")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    history = display.history_for_rewrite(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(display.escape_for_markdown(question))

    with st.chat_message("assistant"):
        result = None
        with st.spinner("Searching the documents..."):
            try:
                result = answer_question(question, history=history, options=options)
            except RuntimeError as error:
                st.error(f"Sorry, the answer service is not available right now. {error}")
        if result is not None:
            reply = display.summarize(result)
            st.session_state.messages.append(reply)
            render_assistant(reply, show_details)