import os
import re
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="RAG PDF Q&A (Hybrid)", page_icon="📘", layout="wide")

# ----------------------------
# Caching: load models once
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2") # fast & good quality
    generator = pipeline("text2text-generation", model="google/flan-t5-large") # better than small, still CPU-friendly
    return embedder, generator

embedder, generator = load_models()

# ----------------------------
# Helpers
# ----------------------------
def clean_text(text: str) -> str:
    """Basic PDF cleanup: remove repeated whitespace and hyphenation artifacts."""
    if not text:
        return ""
    # join broken hyphenated words across line breaks
    text = re.sub(r"-\s+\n", "-", text)
    # replace newlines with spaces
    text = text.replace("\n", " ")
    # collapse multiple spaces
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
    """Sentence/paragraph aware splitter with overlap to preserve context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    return splitter.split_text(text)

@st.cache_data(show_spinner=True)
def build_index(chunks: list[str]):
    """Encode chunks and build FAISS index (cached for the uploaded document)."""
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index, embeddings

def retrieve(index, chunks, query: str, top_k: int = 5):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec.astype("float32"), top_k)
    retrieved = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    return retrieved

def build_prompt(query: str, contexts: list[str], max_context_chars: int = 6000):
    """Concatenate top chunks with a clear instruction to ground the answer."""
    ctx = ""
    for c in contexts:
        if len(ctx) + len(c) + 2 > max_context_chars:
            break
        ctx += c + "\n\n"
    prompt = (
        "You are a helpful assistant answering ONLY using the provided manual excerpts.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{ctx}\n"
        f"Question: {query}\n\n"
        "Answer clearly and concisely in 1–4 sentences."
    )
    return prompt

# ----------------------------
# UI
# ----------------------------
st.title("📘 RAG-based PDF Q&A (Hybrid: FAISS + Flan-T5)")
st.caption("Upload a user manual and ask questions. Uses local embeddings + generative answering.")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size (characters)", 300, 1200, 600, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 300, 120, 10)
    top_k = st.slider("Top-k chunks to retrieve", 1, 8, 5, 1)
    max_out_tokens = st.slider("Max output tokens", 80, 512, 220, 10)
    st.markdown("---")
    st.info("Tip: Bigger chunks + some overlap → better context.\nTop-k=3–5 usually works well.")

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
        with st.spinner("Retrieving and generating answer..."):
            contexts = retrieve(index, chunks, query, top_k=top_k)
            prompt = build_prompt(query, contexts)
            gen = generator(prompt, max_length=max_out_tokens, do_sample=False)
            answer = gen[0]["generated_text"] if gen and len(gen) else "Sorry, I couldn't generate an answer."

        st.subheader("🔎 Answer")
        st.write(answer)

        with st.expander("📄 Sources (retrieved context)"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i}:**\n\n{c[:1000]}{'...' if len(c) > 1000 else ''}")
else:
    st.info("Upload your user manual PDF to begin.")