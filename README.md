# Medical Chatbot

This project is a RAG-powered medical assistant that can answer questions from a knowledge base and generate prescriptions in PDF format.

It includes a FastAPI backend and two separate frontend applications:
1.  **HTML, CSS, JS Frontend**: A modern, responsive chat interface.
2.  **Streamlit Frontend**: A simple, functional chat interface.

---

## Project Structure

```
/
├── backend/                # FastAPI backend
├── html_css_js_frontend/   # Vanilla JS frontend
├── streamlit_frontend/     # Streamlit frontend
└── knowledge/              # Folder for knowledge base PDFs
```

---

## How to Run

### 1. Backend (FastAPI)

The backend serves the chat logic and PDF generation.

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the server:**
    ```bash
    uvicorn main:app --reload
    ```
    The backend will be running at `http://localhost:8000`.

### 2. HTML, CSS, JS Frontend (Recommended)

This is the primary, feature-rich frontend.

1.  **Navigate to the `html_css_js_frontend` directory:**
    ```bash
    cd html_css_js_frontend
    ```
2.  **Run a simple web server:**
    ```bash
    python -m http.server 5500
    ```
3.  **Open your browser** and go to `http://localhost:5500`.

*Note: You can also open the `index.html` file directly in your browser, but using a server is recommended.*


### 3. Streamlit Frontend

This is an alternative, simpler frontend.

1.  **Navigate to the streamlit_frontend directory:**
    ```bash
    cd streamlit_frontend
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    The frontend will be accessible at `http://localhost:8501`.
