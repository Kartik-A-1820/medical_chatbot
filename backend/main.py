import os
import io
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# ======================== ENV / LLM ========================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""
if not os.environ["GOOGLE_API_KEY"]:
    raise RuntimeError("GEMINI_API_KEY not found in environment.")

# LLMs
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
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
qa_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=(
        "You are a cautious medical assistant. Use ONLY the provided context to answer.\n\n"
        "Rules:\n"
        "- Be clear and concise.\n"
        "- You may suggest POSSIBLE causes but avoid definitive diagnoses.\n"
        "- Give step-by-step self-care/remedies when safe.\n"
        "- Call out RED FLAGS that require urgent care.\n"
        "- Prefer generic OTC names when relevant.\n"
        "- Keep tone reassuring but firm about escalation when needed.\n\n"
        "Context:\n{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Question:\n{question}\n\n"
        "Answer (short paragraphs; bullets when useful):"
    ),
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
memory = ConversationBufferWindowMemory(
    k=100, memory_key="chat_history", return_messages=True, output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
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

# ======================== API SCHEMAS ======================
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    references: List[str]

# ======= Structured RX schema (what the LLM must return) ===
class Medicine(BaseModel):
    name: str = Field(description="Name of the medicine.")
    adult: str = Field(default="—", description="Dosage for adults.")
    child: str = Field(default="—", description="Dosage for children.")
    frequency: str = Field(default="", description="How often to take the medicine.")
    duration: str = Field(default="", description="For how long to take the medicine.")
    notes: str = Field(default="", description="Additional notes or warnings.")

class RxSchema(BaseModel):
    possible_disease: str = Field(description="Most likely condition or disease based on the chat.")
    symptoms: List[str] = Field(default_factory=list, description="Key symptoms mentioned by the user.")
    remedies: List[str] = Field(default_factory=list, description="Recommended self-care or diagnostic steps.")
    medicines: List[Medicine] = Field(default_factory=list, description="List of suggested over-the-counter medicines.")
    safety_note: str = Field(default="This is generated by an AI, please consult doctors before following these blindly.", description="A mandatory safety disclaimer.")

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

    # Use .get for safer dictionary access
    doctor_info = payload.get("doctor", {})
    patient_info = payload.get("patient", {})

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(left, y, doctor_info.get("clinic", "Virtual Health"))
    p.setFont("Helvetica", 10)
    p.drawString(left, y - 18, f"{doctor_info.get('name', 'Dr. On-call')}")
    p.drawRightString(right, y, f"Date: {payload.get('date', 'N/A')}")
    y -= 32
    p.line(left, y, right, y)
    y -= 16

    # Patient
    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "Patient Information")
    y -= 16
    p.setFont("Helvetica", 10)
    p.drawString(left, y, f"Name: {patient_info.get('name', 'Patient')}")
    y -= 14
    age_show = patient_info.get('age')
    gender_show = patient_info.get('gender') or "—"
    p.drawString(left, y, f"Age: {age_show if age_show is not None else 'Not specified'}")
    p.drawString(left + 200, y, f"Gender: {gender_show}")
    y -= 22

    # Assessment
    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "Assessment")
    y -= 14
    p.setFont("Helvetica", 10)
    y = draw_wrapped_text(p, left, y, f"Possible Disease: {payload.get('possible_disease', 'Not specified')}", max_width=right-left)
    y -= 6

    # Symptoms
    if payload.get("symptoms"):
        p.setFont("Helvetica-Bold", 12)
        p.drawString(left, y, "Symptoms")
        y -= 14
        p.setFont("Helvetica", 10)
        for s in payload["symptoms"]:
            y = draw_wrapped_text(p, left, y, f"• {s}", max_width=right-left)
            y -= 2
        y -= 6

    # Remedies / Diagnosis / Treatments
    if payload.get("remedies"):
        p.setFont("Helvetica-Bold", 12)
        p.drawString(left, y, "Remedies / Diagnosis / Treatments")
        y -= 14
        p.setFont("Helvetica", 10)
        for r in payload["remedies"]:
            y = draw_wrapped_text(p, left, y, f"• {r}", max_width=right-left)
            y -= 2
        y -= 6

    # Medicines
    if payload.get("medicines"):
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
        result = qa_chain({"question": request.query})
        references = [doc.page_content for doc in result.get("source_documents", [])]
        answer = (result["answer"] or "").strip()

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
# 1. Output Parser
RX_PARSER = PydanticOutputParser(pydantic_object=RxSchema)

# 2. Main Prompt Template
RX_PROMPT_TEMPLATE = (
    "You are an expert at extracting structured data from a conversation. "
    "Your sole job is to prepare a concise, professional prescription summary from the chat. "
    "You MUST ONLY output a single, valid JSON object that conforms to the schema provided.\n\n"
    "Rules:\n"
    "- If age is unknown, ALWAYS include both adult and child guidance in 'medicines'.\n"
    "- Prefer common OTC examples (e.g., Paracetamol/Acetaminophen for fever, ORS for dehydration).\n"
    "- Keep it specific to the query; avoid unrelated diseases.\n"
    "- Keep medicine count small (1–4) with practical dosing.\n"
    "- ALWAYS include the mandatory safety_note.\n\n"
    "Format Instructions:\n"
    "{format_instructions}\n\n"
    "Conversation Context:\n"
    "---------------------\n"
    "User query: {user_query}\n"
    "Assistant answer: {assistant_answer}\n"
    "Patient age (may be None): {age}\n"
    "Patient gender (may be None): {gender}\n"
    "---------------------\n\n"
    "JSON Output:"
)
RX_PROMPT = PromptTemplate(
    input_variables=["user_query", "assistant_answer", "age", "gender"],
    partial_variables={"format_instructions": RX_PARSER.get_format_instructions()},
    template=RX_PROMPT_TEMPLATE,
)

# 3. Error-Fixing Prompt Template
RX_FIX_TEMPLATE = (
    "You previously tried to generate a JSON object but failed. "
    "The error was:\n{error}\n\n"
    "Here is the original, faulty JSON output you provided:\n"
    "```json\n{output}\n```\n\n"
    "Please correct the JSON output based on the error message and the original context. "
    "You MUST ONLY output a single, valid JSON object that conforms to the schema. "
    "Do not add any commentary or explanation.\n\n"
    "Original Context:\n"
    "---------------------\n"
    "User query: {user_query}\n"
    "Assistant answer: {assistant_answer}\n"
    "Patient age (may be None): {age}\n"
    "Patient gender (may be None): {gender}\n"
    "---------------------\n\n"
    "Corrected JSON Output:"
)
RX_FIX_PROMPT = PromptTemplate.from_template(RX_FIX_TEMPLATE)

# 4. Robust JSON Generation Chain with Retries
async def generate_robust_rx_json(inputs: Dict[str, Any], max_retries: int = 2) -> RxSchema:
    """Generates structured JSON, retrying with a 'fixer' prompt on validation errors."""
    # Main chain for the first attempt
    chain = RX_PROMPT | llm | StrOutputParser()

    # Chain for fixing errors
    fix_chain = RX_FIX_PROMPT | llm | StrOutputParser()

    raw_output = await chain.ainvoke(inputs)
    for i in range(max_retries + 1):
        try:
            # Attempt to find and parse a JSON block
            json_match = re.search(r"```json\n({.*?})\n```", raw_output, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(1)
            else:
                # Fallback for when the LLM doesn't use markdown code blocks
                clean_json_str = raw_output.strip()

            parsed_object = RX_PARSER.parse(clean_json_str)
            return parsed_object
        except (ValidationError, json.JSONDecodeError) as e:
            if i >= max_retries:
                raise HTTPException(status_code=500, detail=f"Failed to generate valid JSON after {max_retries} retries. Last error: {e}") from e
            
            # Prepare inputs for the fix chain
            fix_inputs = {**inputs, "error": str(e), "output": raw_output}
            raw_output = await fix_chain.ainvoke(fix_inputs)

    raise HTTPException(status_code=500, detail="Unknown error in JSON generation.")


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
        # Use the new robust generation function
        rx_inputs = {
            "user_query": req.latest_user_query,
            "assistant_answer": req.assistant_answer or "",
            "age": str(req.patient_age) if req.patient_age is not None else "None",
            "gender": req.patient_gender or "None",
        }
        rx: RxSchema = await generate_robust_rx_json(rx_inputs)

        payload = {
            "patient": {"name": req.patient_name or "Patient", "age": req.patient_age, "gender": req.patient_gender},
            "doctor": {"name": req.doctor_name or "Dr. On-call", "clinic": req.clinic_name or "Virtual Health"},
            "date": req.date or datetime.now().strftime("%Y-%m-%d"),
            "possible_disease": rx.possible_disease,
            "symptoms": rx.symptoms,
            "remedies": rx.remedies,
            "medicines": [m.model_dump() for m in rx.medicines],
            "safety_note": rx.safety_note,
        }

        pdf_buffer = create_prescription_pdf(payload)
        return StreamingResponse(pdf_buffer,
                                 media_type="application/pdf",
                                 headers={"Content-Disposition": "attachment; filename=prescription.pdf"})
    except HTTPException as he:
        # Re-raise HTTP exceptions from the generation function
        raise he
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {type(e).__name__}: {e}")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
