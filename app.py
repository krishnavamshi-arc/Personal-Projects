import os
import re
import numpy as np
import faiss
import requests
from flask import Flask, request, render_template, redirect, url_for
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ----------------------------
# Load models once
# ----------------------------
print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Models loaded successfully!")

# ----------------------------
# Helper functions
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"-\s+\n", "-", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    return clean_text(" ".join(pages))

def split_text(text: str, chunk_size=600, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    return splitter.split_text(text)

def build_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index, embeddings

def retrieve(index, chunks, query, top_k=5):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec.astype("float32"), top_k)
    retrieved = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    return retrieved

def build_prompt(query: str, contexts: list[str], max_context_chars=6000):
    ctx = ""
    for c in contexts:
        if len(ctx) + len(c) + 2 > max_context_chars:
            break
        ctx += c + "\n\n"
    prompt = (
        f"{ctx}\n\nQuestion: {query}\n"
        "Answer briefly with explanation, do not restate the question."
    )
    return prompt

def call_google_ai_api(prompt: str, api_url: str, api_key: str):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    response = requests.post(api_url, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "Error parsing API response."
    else:
        return f"API Error: {response.status_code} - {response.text}"

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "pdf" not in request.files:
            return render_template("index.html", error="No file uploaded.")
        file = request.files["pdf"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")
        
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        text = read_pdf(file_path)
        if not text:
            return render_template("index.html", error="Could not extract text from PDF.")

        chunks = split_text(text)
        if len(chunks) == 0:
            return render_template("index.html", error="No text chunks created from PDF.")
        
        index, _ = build_index(chunks)
        return render_template("question.html", chunks_count=len(chunks), file=file.filename)

    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query")
    file_name = request.form.get("file_name")
    api_url = request.form.get("api_url")
    api_key = request.form.get("api_key")

    if not all([query, file_name, api_url, api_key]):
        return render_template("question.html", error="Missing inputs.", file=file_name)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
    text = read_pdf(file_path)
    chunks = split_text(text)
    index, _ = build_index(chunks)

    contexts = retrieve(index, chunks, query, top_k=5)
    prompt = build_prompt(query, contexts)
    answer = call_google_ai_api(prompt, api_url, api_key)

    return render_template("answer.html", answer=answer, contexts=contexts[:5], query=query)

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
