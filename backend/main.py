

import os
import io
from datetime import datetime

# from qtpy import API
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch


# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    references: List[str]

class PrescriptionData(BaseModel):
    patient_name: str = "John Doe"
    patient_age: int = 30
    # ... (rest of the model is unchanged)
    patient_gender: str = "Male"
    doctor_name: str = "Dr. Smith"
    clinic_name: str = "Health Clinic"
    date: str = datetime.now().strftime("%Y-%m-%d")
    disease: str
    symptoms: str
    prescription: str
    normal_values: Optional[str] = Field(None)
    patient_values: Optional[str] = Field(None)


# --- FastAPI App ---

app = FastAPI(title="Medical Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI & DB Setup ---

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

CHROMA_DB_PATH = "chroma_db"
KNOWLEDGE_DIR = "knowledge"


if not os.path.exists("processed_files.txt"):
    with open("processed_files.txt", "w") as f:
        f.write("")  # Initialize the file if it doesn't exist

with open("processed_files.txt", "r") as f:
    processed_files = f.read().splitlines()

if not os.path.exists(CHROMA_DB_PATH):
    print("Database not found. Creating new one from PDF files...")
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        print(f"'{KNOWLEDGE_DIR}' directory created. Please add your PDF files here and restart.")
        # Exit or handle gracefully if no PDFs are found
        all_docs = []
    else:
        pdf_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            print(f"No PDF files found in the '{KNOWLEDGE_DIR}' directory. The chatbot will have no knowledge base.")
            all_docs = []
        else:
            print(f"Found {len(pdf_files)} PDF(s) to process.")
            all_docs = []
            for pdf_file in pdf_files:
                if pdf_file in processed_files:
                    continue
            
            with open("processed_files.txt", "w") as f:
                f.write(pdf_file + "\n")     

            loader = PyPDFLoader(os.path.join(KNOWLEDGE_DIR, pdf_file))
            all_docs.extend(loader.load())

    if all_docs:
        text_splitter = SemanticChunker(embeddings)
        texts = text_splitter.split_documents(all_docs)
        print(f"Splitting documents into {len(texts)} semantic chunks.")
        vector_store = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)
        print("Database created successfully.")
    else:
        # Create an empty store if no documents are present
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

else:
    print("Loading existing vector database.")
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant. Answer questions based on the provided documents."),
    ("human", "{question}"),
    ("assistant", "{answer}")
])

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
memory = ConversationBufferWindowMemory(k=100, memory_key="chat_history", return_messages=True, output_key='answer')
qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# --- PDF Generation (function is unchanged) ---
def create_prescription_pdf(data: PrescriptionData):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # --- Header ---
    p.setFont("Helvetica-Bold", 16)
    p.drawString(inch, height - inch, data.clinic_name)
    p.setFont("Helvetica", 10)
    p.drawString(inch, height - inch - 20, f"Dr. {data.doctor_name}")
    
    p.setFont("Helvetica", 10)
    p.drawRightString(width - inch, height - inch, f"Date: {data.date}")

    p.line(inch, height - inch - 40, width - inch, height - inch - 40)

    # --- Patient Info ---
    p.setFont("Helvetica-Bold", 12)
    p.drawString(inch, height - inch - 60, "Patient Information")
    p.setFont("Helvetica", 10)
    p.drawString(inch, height - inch - 80, f"Name: {data.patient_name}")
    p.drawString(inch, height - inch - 95, f"Age: {data.patient_age}")
    p.drawString(inch, height - inch - 110, f"Gender: {data.patient_gender}")

    # --- Diagnosis Section ---
    y_pos = height - 2.5 * inch
    p.setFont("Helvetica-Bold", 12)
    p.drawString(inch, y_pos, "Diagnosis")
    p.line(inch, y_pos - 10, width - inch, y_pos - 10)
    
    p.setFont("Helvetica", 10)
    p.drawString(inch, y_pos - 30, f"Disease: {data.disease}")
    p.drawString(inch, y_pos - 50, f"Symptoms: {data.symptoms}")

    # --- Vitals/Reports ---
    if data.normal_values or data.patient_values:
        y_pos -= 1.2 * inch
        p.setFont("Helvetica-Bold", 12)
        p.drawString(inch, y_pos, "Diagnostic Values")
        p.line(inch, y_pos - 10, width - inch, y_pos - 10)
        p.setFont("Helvetica", 10)
        y_pos -= 30
        if data.normal_values:
            p.drawString(inch, y_pos, f"Normal Values: {data.normal_values}")
            y_pos -= 20
        if data.patient_values:
            p.drawString(inch, y_pos, f"Patient Report Values: {data.patient_values}")

    # --- Prescription (Rx) ---
    y_pos -= 1.2 * inch
    p.setFont("Helvetica-Bold", 20)
    p.drawString(inch, y_pos, "Rx")
    p.line(inch, y_pos - 15, width - inch, y_pos - 15)
    
    p.setFont("Helvetica", 10)
    text = p.beginText(inch, y_pos - 40)
    text.setLeading(14)
    for line in data.prescription.split('\n'):
        text.textLine(line)
    p.drawText(text)

    # --- Footer ---
    p.setFont("Helvetica-Oblique", 9)
    p.drawCentredString(width / 2.0, inch / 2.0, "This prescription was generated electronically.")

    p.showPage()
    p.save()
    buffer.seek(0);
    return buffer
# --- API Endpoints (are unchanged) ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = qa_chain({"question": request.query})
    references = [doc.page_content for doc in result.get("source_documents", [])]
    return ChatResponse(answer=result["answer"], references=references)

@app.post("/generate_prescription_pdf")
async def generate_prescription_pdf(data: PrescriptionData):
    pdf_buffer = create_prescription_pdf(data)
    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=prescription.pdf"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
