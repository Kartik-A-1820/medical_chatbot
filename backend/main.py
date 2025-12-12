import os
import io
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# ======================== ENV / LLM ========================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""
if not os.environ["GOOGLE_API_KEY"]:
    raise RuntimeError("GEMINI_API_KEY not found in environment.")

# LLMs
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ======================== FASTAPI ==========================
app = FastAPI(title="Medical Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== DB / RAG =========================
CHROMA_DB_PATH = "chroma_db"
KNOWLEDGE_DIR = "knowledge"
PROCESSED_TRACKER = "processed_files.txt"


def build_or_load_vectorstore() -> Chroma:
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    processed = set()
    if os.path.exists(PROCESSED_TRACKER):
        with open(PROCESSED_TRACKER, "r", encoding="utf-8") as f:
            processed = set(l.strip() for l in f if l.strip())

    pdf_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith(".pdf")]

    def split_docs(docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n• ", "\n- ", "\n", ". ", " "]
        )
        return splitter.split_documents(docs)

    if not os.path.exists(CHROMA_DB_PATH):
        all_docs = []
        for pdf in pdf_files:
            if pdf in processed:
                continue
            loader = PyPDFLoader(os.path.join(KNOWLEDGE_DIR, pdf))
            all_docs.extend(loader.load())
            processed.add(pdf)

        if all_docs:
            texts = split_docs(all_docs)
            store = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)
        else:
            store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    else:
        store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        new_docs = []
        for pdf in pdf_files:
            if pdf in processed:
                continue
            loader = PyPDFLoader(os.path.join(KNOWLEDGE_DIR, pdf))
            new_docs.extend(loader.load())
            processed.add(pdf)
        if new_docs:
            texts = split_docs(new_docs)
            store.add_documents(texts)
            store.persist()

    with open(PROCESSED_TRACKER, "w", encoding="utf-8") as f:
        for name in sorted(processed):
            f.write(name + "\n")

    return store


vector_store = build_or_load_vectorstore()

# ======================== SAFETY PROMPT ====================
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.3},
)

chat_history_store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]


contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, rewrite the question "
            "so it is standalone. If it is already standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a cautious medical assistant. Use ONLY the provided context to answer.\n\n"
            "Rules:\n"
            "- Be clear and concise.\n"
            "- You may suggest POSSIBLE causes but avoid definitive diagnoses.\n"
            "- Give step-by-step self-care/remedies when safe.\n"
            "- Call out RED FLAGS that require urgent care.\n"
            "- Prefer generic OTC names when relevant.\n"
            "- Keep tone reassuring but firm about escalation when needed.\n\n"
            "Context:\n{context}\n",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

qa_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ======================== TRIAGE ===========================
RED_FLAG_PATTERNS = [
    "severe chest pain", "pressure in chest", "shortness of breath", "difficulty breathing",
    "blue lips", "unconscious", "fainted", "confusion", "seizure", "stiff neck",
    "worst headache of my life", "sudden severe headache",
    "weakness on one side", "slurred speech", "facial droop",
    "uncontrolled bleeding", "vomiting blood", "black stools",
    "high fever", "fever above 104", "fever > 104", "fever more than 3 days",
    "infant", "baby", "child under 3 months with fever",
    "severe dehydration", "no urination", "sunken eyes", "lethargic",
    "severe abdominal pain", "rigid abdomen",
    "allergic reaction", "swelling of tongue", "throat closing",
]

def detect_emergency(text: str) -> Optional[str]:
    t = (text or "").lower()
    for p in RED_FLAG_PATTERNS:
        if p in t:
            return (
                "🚨 **Potential emergency detected.** Please seek urgent in-person care.\n"
                "If you suspect a life-threatening emergency, call your local emergency number immediately."
            )
    return None


# =============== Lightweight intent heuristics =============
GREETINGS = {"hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "good afternoon"}
SMALL_TALK = {"how are you", "what's up", "whats up", "how is it going", "how's it going", "who are you"}
MEDICAL_KEYWORDS = {
    "pain", "fever", "cough", "symptom", "symptoms", "medicine", "medication", "diagnosis",
    "treatment", "disease", "infection", "rash", "headache", "cold", "flu", "injury",
    "bleeding", "asthma", "diabetes", "hypertension", "allergy", "allergic", "vomit",
    "vomiting", "nausea", "diarrhea", "covid", "corona", "breath", "breathing", "sore throat",
    "migraine", "ache", "fracture", "sprain", "burn", "wound",
}


def is_greeting(text: str) -> bool:
    low = (text or "").strip().lower()
    return any(low == g or low.startswith(g + " ") for g in GREETINGS)


def is_small_talk(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in SMALL_TALK)


def is_medical_query(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in MEDICAL_KEYWORDS)

# ======================== API SCHEMAS ======================
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    references: List[str]

# ======= Structured RX schema (what the LLM must return) ===
class Medicine(BaseModel):
    name: str
    adult: str = "—"
    child: str = "—"
    frequency: str = ""
    duration: str = ""
    notes: str = ""

class RxSchema(BaseModel):
    possible_disease: str
    symptoms: List[str] = Field(default_factory=list)
    remedies: List[str] = Field(default_factory=list)
    medicines: List[Medicine] = Field(default_factory=list)
    # default ensures presence even if model omits it
    safety_note: str = "This is generated by an AI, please consult doctors before following these blindly."

# =================== PDF LAYOUT (ReportLab) =================
def draw_wrapped_text(p, x, y, text, max_width, leading=14, font="Helvetica", size=10):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    p.setFont(font, size)
    words = text.split()
    line = ""
    lines = []
    for w in words:
        test = (line + " " + w).strip()
        if stringWidth(test, font, size) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    for ln in lines:
        p.drawString(x, y, ln)
        y -= leading
    return y

def create_prescription_pdf(payload: Dict[str, Any]) -> io.BytesIO:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left = inch
    right = width - inch
    y = height - inch

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(left, y, payload["doctor"]["clinic"])
    p.setFont("Helvetica", 10)
    p.drawString(left, y - 18, f"{payload['doctor']['name']}")
    p.drawRightString(right, y, f"Date: {payload['date']}")
    y -= 32
    p.line(left, y, right, y)
    y -= 16

    # Patient
    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "Patient Information")
    y -= 16
    p.setFont("Helvetica", 10)
    p.drawString(left, y, f"Name: {payload['patient']['name']}")
    y -= 14
    age_show = payload['patient'].get('age')
    gender_show = payload['patient'].get('gender') or "—"
    p.drawString(left, y, f"Age: {age_show if age_show is not None else 'Not specified'}")
    p.drawString(left + 200, y, f"Gender: {gender_show}")
    y -= 22

    # Assessment
    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "Assessment")
    y -= 14
    p.setFont("Helvetica", 10)
    y = draw_wrapped_text(p, left, y, f"Possible Disease: {payload['possible_disease']}", max_width=right-left)
    y -= 6

    # Symptoms
    if payload["symptoms"]:
        p.setFont("Helvetica-Bold", 12)
        p.drawString(left, y, "Symptoms")
        y -= 14
        p.setFont("Helvetica", 10)
        for s in payload["symptoms"]:
            y = draw_wrapped_text(p, left, y, f"• {s}", max_width=right-left)
            y -= 2
        y -= 6

    # Remedies / Diagnosis / Treatments
    if payload["remedies"]:
        p.setFont("Helvetica-Bold", 12)
        p.drawString(left, y, "Remedies / Diagnosis / Treatments")
        y -= 14
        p.setFont("Helvetica", 10)
        for r in payload["remedies"]:
            y = draw_wrapped_text(p, left, y, f"• {r}", max_width=right-left)
            y -= 2
        y -= 6

    # Medicines
    if payload["medicines"]:
        p.setFont("Helvetica-Bold", 12)
        p.drawString(left, y, "Medicines")
        y -= 14
        p.setFont("Helvetica", 10)
        for m in payload["medicines"]:
            name = m.get("name", "Medicine")
            adult = m.get("adult", "—")
            child = m.get("child", "—")
            freq = m.get("frequency", "")
            dur = m.get("duration", "")
            notes = m.get("notes", "")
            y = draw_wrapped_text(
                p, left, y,
                f"• {name} — Adult: {adult} | Child: {child}"
                + (f" | Frequency: {freq}" if freq else "")
                + (f" | Duration: {dur}" if dur else ""),
                max_width=right-left
            )
            if notes:
                y = draw_wrapped_text(p, left + 14, y, f"Note: {notes}", max_width=right-left-14)
            y -= 4

    # Safety Note
    y -= 8
    p.setFont("Helvetica-Oblique", 9)
    y = draw_wrapped_text(
        p, left, y,
        payload.get("safety_note", "This is generated by an AI; please consult a doctor before following."),
        max_width=right-left, leading=12, font="Helvetica-Oblique", size=9
    )

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ======================== ENDPOINTS ========================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        query = (request.query or "").strip()

        if is_greeting(query):
            answer = (
                "Hello! I'm a medical assistant focused on health questions about symptoms, "
                "conditions, and self-care. How can I help today?"
            )
            return ChatResponse(answer=answer, references=[])

        if is_small_talk(query) or not is_medical_query(query):
            answer = (
                "I'm here to discuss health concerns, symptoms, medications, and related topics. "
                "Feel free to share any medical question, and I'll do my best to help."
            )
            return ChatResponse(answer=answer, references=[])

        result = await qa_chain.ainvoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id or "default"}},
        )
        references = [doc.page_content for doc in result.get("context", [])]
        answer = (result.get("answer") or "").strip()

        # Rules-based triage on both user text and answer
        triage = detect_emergency(request.query) or detect_emergency(answer)
        if triage:
            answer += ("\n\n---\n" + triage)

        # Safety nudge
        answer += (
            "\n\n---\n**Note:** I’m not a substitute for a doctor. "
            "If symptoms are severe, new, or worsening — or you suspect an emergency — seek in-person care."
        )
        return ChatResponse(answer=answer, references=references)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

# ======= Structured-output prompt & chain for RX ===========
RX_PARSER = PydanticOutputParser(pydantic_object=RxSchema)

RX_PROMPT = PromptTemplate(
    input_variables=["user_query", "assistant_answer", "age", "gender"],
    partial_variables={"format_instructions": RX_PARSER.get_format_instructions()},
    template=(
        "Prepare a concise, professional prescription summary from the chat.\n"
        "- If age is unknown, ALWAYS include both adult and child guidance in 'medicines'.\n"
        "- Prefer common OTC examples (e.g., Paracetamol/Acetaminophen for fever, ORS for dehydration).\n"
        "- Keep it specific to the query; avoid unrelated diseases.\n"
        "- Keep medicine count small (1–4) with practical dosing.\n"
        "- ALWAYS include this sentence in safety_note: "
        "'This is generated by an AI, please consult doctors before following these blindly.'\n\n"
        "{format_instructions}\n\n"
        "User query: {user_query}\n"
        "Assistant answer: {assistant_answer}\n"
        "Patient age (may be None): {age}\n"
        "Patient gender (may be None): {gender}\n"
    ),
)

RX_CHAIN = RX_PROMPT | llm | StrOutputParser() | RX_PARSER

class RxFromChatRequest(BaseModel):
    latest_user_query: str
    assistant_answer: Optional[str] = None
    patient_name: Optional[str] = "Patient"
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    doctor_name: Optional[str] = "Dr. On-call"
    clinic_name: Optional[str] = "Virtual Health"
    date: Optional[str] = datetime.now().strftime("%Y-%m-%d")

@app.post("/generate_prescription_pdf_from_chat")
async def generate_prescription_pdf_from_chat(req: RxFromChatRequest):
    try:
        rx: RxSchema = await RX_CHAIN.ainvoke({
            "user_query": req.latest_user_query,
            "assistant_answer": req.assistant_answer or "",
            "age": str(req.patient_age) if req.patient_age is not None else "None",
            "gender": req.patient_gender or "None",
        })

        payload = {
            "patient": {"name": req.patient_name or "Patient", "age": req.patient_age, "gender": req.patient_gender},
            "doctor": {"name": req.doctor_name or "Dr. On-call", "clinic": req.clinic_name or "Virtual Health"},
            "date": req.date or datetime.now().strftime("%Y-%m-%d"),
            "possible_disease": rx.possible_disease,
            "symptoms": rx.symptoms,
            "remedies": rx.remedies,
            "medicines": [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in rx.medicines],
            "safety_note": rx.safety_note,
        }

        pdf_buffer = create_prescription_pdf(payload)
        return StreamingResponse(pdf_buffer,
                                 media_type="application/pdf",
                                 headers={"Content-Disposition": "attachment; filename=prescription.pdf"})
    except ValidationError as ve:
        raise HTTPException(status_code=500, detail=f"Model returned invalid schema: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {type(e).__name__}: {e}")

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
