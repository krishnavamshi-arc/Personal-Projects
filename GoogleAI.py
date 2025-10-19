import os
import re
import json
import requests
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="RAG PDF Q&A (Hybrid with Google AI API)", page_icon="📘", layout="wide")

# ----------------------------
# API Configuration
# ----------------------------
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"   # 🔹 Replace with your actual endpoint
API_KEY = "AIzaSyAAlV9eAGcg4yShhU0o6CE0-cFKPV8FsnY"   # or store securely in Streamlit secrets

# ----------------------------
# Caching: load embedding model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ----------------------------
# Helpers
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"-\s+\n", "-", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def read_pdf(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    return clean_text(" ".join(pages))

def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    return splitter.split_text(text)

@st.cache_data(show_spinner=True)
def build_index(chunks: list[str]):
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings.astype("float32"))
    return index, embeddings

def retrieve(index, chunks, query: str, top_k: int = 5):
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_vec.astype("float32"), top_k)
    retrieved = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    return retrieved

def build_prompt(query: str, contexts: list[str], max_context_chars: int = 6000):
    ctx = ""
    for c in contexts:
        if len(ctx) + len(c) + 2 > max_context_chars:
            break
        ctx += c + "\n\n"
    # Include both question and context for grounding
    prompt = (
        f"Using the information below, provide a factual and concise answer with explanation.\n"
        f"Context:\n{ctx}\n"
        f"Question: {query}\n\n"
        "Answer briefly and clearly."
    )
    return prompt

def call_google_ai_api(prompt: str):
    """Call Google AI REST API with your prompt."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            result = response.json()
            # Extract model output (adjust based on API response format)
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response text.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception while calling API: {e}"

# ----------------------------
# UI
# ----------------------------
st.title("📘 RAG-based PDF Q&A (Google AI API)")
st.caption("Upload a PDF manual and ask questions. Uses FAISS retrieval + Google AI generation API.")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size (characters)", 300, 1200, 600, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 300, 120, 10)
    top_k = st.slider("Top-k chunks to retrieve", 1, 8, 5, 1)
    st.markdown("---")
    st.info("Tune chunk size and overlap for best retrieval quality.")

pdf = st.file_uploader("Upload a PDF manual", type=["pdf"])

if pdf:
    with st.spinner("Reading & processing PDF..."):
        text = read_pdf(pdf)
        if not text:
            st.error("Could not extract text from the PDF. Try another file.")
            st.stop()
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if len(chunks) == 0:
            st.error("No text chunks were created. Try reducing chunk size or using a different PDF.")
            st.stop()
        index, _ = build_index(chunks)

    st.success(f"Indexed {len(chunks)} chunks. Ask your question below.")
    query = st.text_input("Your question")

    if query:
        with st.spinner("Retrieving and calling Google AI API..."):
            contexts = retrieve(index, chunks, query, top_k=top_k)
            prompt = build_prompt(query, contexts)
            answer = call_google_ai_api(prompt)

        st.subheader("🔎 Answer")
        st.write(answer)

        with st.expander("📄 Sources (retrieved context)"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i}:**\n\n{c[:1000]}{'...' if len(c) > 1000 else ''}")
else:
    st.info("Upload your user manual PDF to begin.")
