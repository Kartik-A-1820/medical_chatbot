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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7 )
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

CHROMA_DB_PATH = "chroma_db"
KNOWLEDGE_DIR = "knowledge"

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
