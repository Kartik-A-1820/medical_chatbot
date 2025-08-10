# app.py
import os
import io
import json
import requests
import streamlit as st
from datetime import datetime

# ============== CONFIG ==============
# Point this to your FastAPI server
BACKEND_URL = os.getenv("MEDCHAT_BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== STYLES ==============
CHAT_STYLE = """
<style>
/* Tighter top padding for a ChatGPT-like compact look */
section.main > div { padding-top: 1rem; }

/* Make chat bubbles a bit more card-like */
.chat-bubble {
  border-radius: 12px;
  padding: 12px 14px;
  margin: 6px 0;
  border: 1px solid rgba(120,120,120,0.15);
  background: rgba(30,30,30,0.5);
}
[data-theme="dark"] .chat-bubble {
  background: rgba(20,20,20,0.35);
  border-color: rgba(255,255,255,0.08);
}

/* Monospace for references for readability */
.refs pre {
  white-space: pre-wrap;
  font-size: 0.86rem;
}

/* Make the input area stick to bottom like ChatGPT */
.block-container { padding-bottom: 4rem; }
</style>
"""
st.markdown(CHAT_STYLE, unsafe_allow_html=True)

# ============== STATE ==============
if "messages" not in st.session_state:
    # Each message: {"role": "user"/"assistant", "content": str, "references": Optional[List[str]]}
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I’m your medical assistant. Ask me about your reports or symptoms. "
                       "I’ll answer based on the knowledge base and add cautions. "
                       "For emergencies or critical concerns, please consult a doctor.",
            "references": None,
        }
    ]

if "last_refs" not in st.session_state:
    st.session_state.last_refs = None

# ============== HELPERS ==============
def post_chat(query: str):
    """Call FastAPI /chat endpoint."""
    url = f"{BACKEND_URL}/chat"
    try:
        resp = requests.post(url, json={"query": query}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        references = data.get("references", [])
        return answer, references, None
    except requests.RequestException as e:
        return None, None, f"Backend error: {str(e)}"

def post_generate_pdf(payload: dict):
    """Call FastAPI /generate_prescription_pdf endpoint and return bytes."""
    url = f"{BACKEND_URL}/generate_prescription_pdf"
    try:
        resp = requests.post(url, json=payload, timeout=120, stream=True)
        resp.raise_for_status()
        return resp.content, None
    except requests.RequestException as e:
        return None, f"PDF generation failed: {str(e)}"

def chat_message(role, content, references=None):
    avatar = "🧑‍⚕️" if role == "assistant" else "🧑‍💻"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f'<div class="chat-bubble">{content}</div>', unsafe_allow_html=True)
        if references:
            with st.expander("Sources / Extracted References"):
                # Show refs in a compact way
                for i, ref in enumerate(references, 1):
                    st.markdown(f"**Source {i}**")
                    st.code(ref)

# ============== SIDEBAR ==============
with st.sidebar:
    st.subheader("🩺 Prescription Generator")
    with st.form("rx_form", clear_on_submit=False):
        colA, colB = st.columns(2)
        with colA:
            patient_name = st.text_input("Patient Name", "John Doe")
            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30, step=1)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
        with colB:
            doctor_name = st.text_input("Doctor Name", "Dr. Smith")
            clinic_name = st.text_input("Clinic Name", "Health Clinic")
            date_str = st.text_input("Date (YYYY-MM-DD)", datetime.now().strftime("%Y-%m-%d"))

        disease = st.text_input("Disease", placeholder="e.g., Acute Pharyngitis")
        symptoms = st.text_area("Symptoms", height=80, placeholder="e.g., fever, sore throat, cough")
        normal_values = st.text_area("Normal Values (optional)", height=80, placeholder="e.g., CBC normal ranges")
        patient_values = st.text_area("Patient Report Values (optional)", height=80, placeholder="e.g., WBC 12k/µL, CRP 22 mg/L")

        prescription = st.text_area(
            "Prescription (Final Rx Text)",
            height=160,
            placeholder=(
                "Tab. Paracetamol 500 mg – 1 tab q6h PRN fever (max 4/day)\n"
                "Tab. Levocetirizine 5 mg – 1 tab HS × 5 days\n"
                "Salt-water gargle TID × 3 days\n"
                "Hydration, rest. Re-evaluate if fever > 48h or red flags (breathing difficulty, confusion)."
            )
        )

        gen_col1, gen_col2 = st.columns([1, 1])
        with gen_col1:
            gen_btn = st.form_submit_button("Generate Prescription PDF", use_container_width=True)
        with gen_col2:
            clear_btn = st.form_submit_button("Clear Chat", use_container_width=True)

    if clear_btn:
        st.session_state.messages = st.session_state.messages[:1]  # keep only the greeting
        st.session_state.last_refs = None
        st.rerun()

    if gen_btn:
        payload = {
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "doctor_name": doctor_name,
            "clinic_name": clinic_name,
            "date": date_str,
            "disease": disease,
            "symptoms": symptoms,
            "prescription": prescription,
            "normal_values": normal_values if normal_values.strip() else None,
            "patient_values": patient_values if patient_values.strip() else None,
        }
        with st.spinner("Generating PDF..."):
            pdf_bytes, err = post_generate_pdf(payload)
        if err:
            st.error(err)
        else:
            st.success("Prescription generated.")
            st.download_button(
                label="Download prescription.pdf",
                data=pdf_bytes,
                file_name="prescription.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.divider()
    st.caption("Backend URL")
    st.code(BACKEND_URL, language="bash")
    st.caption("Tip: set env var MEDCHAT_BACKEND_URL to point to a remote server.")

# ============== MAIN CHAT AREA ==============
st.title("🩺 Medical Chatbot")
st.caption("RAG-powered assistant. For critical or emergency issues, consult a medical professional.")

# show history
for msg in st.session_state.messages:
    chat_message(msg["role"], msg["content"], msg.get("references"))

# chat input (like ChatGPT)
if user_prompt := st.chat_input("Ask a medical question…"):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_prompt, "references": None})
    chat_message("user", user_prompt)

    # Call backend and display assistant response
    with st.spinner("Thinking…"):
        answer, refs, err = post_chat(user_prompt)

    if err:
        assistant_text = (
            "Sorry, I couldn’t reach the backend right now.\n\n"
            f"```\n{err}\n```"
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_text, "references": None})
        chat_message("assistant", assistant_text)
    else:
        # Add safety nudge automatically
        caution = (
            "\n\n---\n**Note:** I’m not a substitute for a doctor. "
            "If your symptoms are severe, new, or worsening — or if you suspect a medical emergency — "
            "seek in-person care promptly."
        )
        full_answer = (answer or "").strip() + caution
        st.session_state.messages.append({"role": "assistant", "content": full_answer, "references": refs or None})
        st.session_state.last_refs = refs or None
        chat_message("assistant", full_answer, refs or None)
