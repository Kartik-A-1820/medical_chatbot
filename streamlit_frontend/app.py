import os
import requests
import streamlit as st
import streamlit.components.v1 as components  # for auto-scroll

BACKEND_URL = os.getenv("MEDCHAT_BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CHAT_STYLE = """
<style>
section.main > div { padding-top: 1rem; }
.chat-bubble { border-radius: 12px; padding: 12px 14px; margin: 6px 0;
  border: 1px solid rgba(120,120,120,0.15); background: rgba(30,30,30,0.5); }
[data-theme="dark"] .chat-bubble { background: rgba(20,20,20,0.35); border-color: rgba(255,255,255,0.08); }
.refs pre { white-space: pre-wrap; font-size: 0.86rem; }
.block-container { padding-bottom: 6rem; }
</style>
"""
st.markdown(CHAT_STYLE, unsafe_allow_html=True)

# ----------- State -----------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hi! I’m your medical assistant. Ask me about your reports or symptoms.\n"
            "I’ll use the knowledge base and include cautions. For emergencies, consult a doctor immediately."
        ),
        "references": None,
    }]

if "needs_scroll" not in st.session_state:
    st.session_state.needs_scroll = False

if "last_pdf_bytes" not in st.session_state:
    st.session_state.last_pdf_bytes = None

# ----------- Backend calls -----------
def post_chat(query: str):
    url = f"{BACKEND_URL}/chat"
    resp = requests.post(url, json={"query": query}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", ""), data.get("references", [])

def post_rx_from_chat(latest_user_query: str, assistant_answer: str|None):
    url = f"{BACKEND_URL}/generate_prescription_pdf_from_chat"
    payload = {
        "latest_user_query": latest_user_query,
        "assistant_answer": assistant_answer or "",
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.content  # PDF bytes

# ----------- UI helpers -----------
def chat_message(role, content, references=None):
    avatar = "🧑‍⚕️" if role == "assistant" else "🧑‍💻"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f'<div class="chat-bubble">{content}</div>', unsafe_allow_html=True)
        if references:
            with st.expander("Sources"):
                for i, ref in enumerate(references, 1):
                    st.markdown(f"**Source {i}**")
                    st.code(ref)

def scroll_to_bottom():
    components.html(
        """<script>window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'});</script>""",
        height=0,
    )

def generate_pdf_flow():
    last_user = next((m for m in reversed(st.session_state.messages) if m["role"]=="user"), None)
    last_assistant = next((m for m in reversed(st.session_state.messages) if m["role"]=="assistant"), None)
    if not last_user or not last_assistant:
        st.warning("Ask a question first so I can base the PDF on it.")
        return
    with st.spinner("Generating PDF..."):
        try:
            pdf_bytes = post_rx_from_chat(last_user["content"], last_assistant["content"])
        except requests.RequestException as e:
            st.error(f"PDF generation failed: {e}")
            return
    st.session_state.last_pdf_bytes = pdf_bytes
    st.success("Prescription generated. Use the download button below.")

# ----------- Header (top) with PDF button -----------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🩺 Medical Chatbot")
    st.caption("RAG-powered. For critical or emergency issues, consult a medical professional.")
with col2:
    st.write("")
    st.write("")
    if st.button("Generate PDF from this chat", use_container_width=True, key="btn_pdf_top"):
        generate_pdf_flow()

# ----------- Conversation history -----------
for m in st.session_state.messages:
    chat_message(m["role"], m["content"], m.get("references"))

st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

# ----------- Bottom action bar -----------
st.divider()
bcol1, bcol2 = st.columns([0.6, 0.4])
with bcol1:
    st.caption("Tip: Include age/weight if you want age-specific guidance. The PDF always includes a safety note.")
with bcol2:
    c1, c2 = st.columns(2)
    if c1.button("Generate PDF", use_container_width=True, key="btn_pdf_bottom"):
        generate_pdf_flow()
    if st.session_state.last_pdf_bytes:
        c2.download_button(
            label="Download prescription.pdf",
            data=st.session_state.last_pdf_bytes,
            file_name="prescription.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_pdf_bottom"
        )

# ----------- Chat input -----------
if user_prompt := st.chat_input("Describe your symptoms or ask about your report…"):
    st.session_state.messages.append({"role": "user", "content": user_prompt, "references": None})
    st.session_state.needs_scroll = True
    chat_message("user", user_prompt)

    with st.spinner("Thinking…"):
        try:
            answer, refs = post_chat(user_prompt)
        except requests.RequestException as e:
            assistant_text = f"Sorry, I couldn’t reach the backend.\n\n```\n{e}\n```"
            st.session_state.messages.append({"role": "assistant", "content": assistant_text, "references": None})
            chat_message("assistant", assistant_text)
        else:
            full_answer = (answer or "").strip()
            st.session_state.messages.append({"role": "assistant", "content": full_answer, "references": refs or None})
            chat_message("assistant", full_answer, refs or None)
    st.session_state.needs_scroll = True

# ----------- Auto-scroll -----------
if st.session_state.needs_scroll:
    scroll_to_bottom()
    st.session_state.needs_scroll = False
