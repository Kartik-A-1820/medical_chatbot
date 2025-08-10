# medical_chatbot

This project is a medical chatbot with a RAG-powered assistant. It can answer medical questions based on a knowledge base of PDF documents. It also includes a prescription generator.

## Running the Application

The application consists of a backend server and a frontend client.

### Backend

The backend is a FastAPI application. To run it, follow these steps:

1.  **Navigate to the `backend` directory:**
    ```bash
    cd backend
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```
    The server will be running at `http://localhost:8000`.

### Frontend

The frontend is a Streamlit application. To run it, follow these steps:

1.  **Navigate to the `frontend` directory:**
    ```bash
    cd frontend
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser at `http://localhost:8501`.
