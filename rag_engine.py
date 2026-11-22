# rag_engine.py
import os
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# override=True makes the key in .env take priority over old Windows variables
load_dotenv(override=True)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-20b"        # Groq chat model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"    # free local embedding model

SYSTEM_PROMPT = """You are a helpful assistant that answers questions
using ONLY the context provided from a document.
- If the answer is not in the context, say:
  "I could not find this in the document."
- Mention page numbers in your answer, like (page 3).
- Be clear, accurate, and concise."""

_embedder = None  # the embedding model is loaded only once


def get_client() -> OpenAI:
    """Create the Groq client (Groq uses the OpenAI-compatible API)."""
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Check that your .env file is named "
            ".env (not .env.txt) and is in the project folder."
        )
    if not api_key.startswith("gsk_"):
        raise RuntimeError(
            "This does not look like a Groq key. Groq keys start with 'gsk_'. "
            "Create one at console.groq.com and put it in .env."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def get_embedder() -> SentenceTransformer:
    """Load the local embedding model once and reuse it."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# ---------- 1. Read the PDF page by page ----------
def read_pdf(file) -> list[dict]:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


# ---------- 2. Split text into overlapping chunks ----------
def chunk_pages(pages, chunk_size=800, overlap=150) -> list[dict]:
    chunks = []
    for p in pages:
        text = " ".join(p["text"].split())  # remove extra whitespace
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"page": p["page"], "text": text[start:end]})
            start = end - overlap  # overlap keeps sentences from being cut off
    return chunks


# ---------- 3. Convert text into vectors (embeddings) ----------
def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,  # normalized vectors: dot product = cosine similarity
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


# ---------- 4. Build the index for one PDF ----------
def build_index(file) -> dict:
    pages = read_pdf(file)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No text found in this PDF. It may be a scanned image.")
    embeddings = embed_texts([c["text"] for c in chunks])
    return {"chunks": chunks, "embeddings": embeddings}


# ---------- 5. Find the chunks most similar to the question ----------
def retrieve(question: str, index: dict, top_k: int = 4) -> list[dict]:
    q = embed_texts([question])[0]
    matrix = index["embeddings"]
    sims = matrix @ q  # vectors are normalized, so this is cosine similarity

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        hit = dict(index["chunks"][i])
        hit["score"] = float(sims[i])
        results.append(hit)
    return results


# ---------- 6. Send the chunks and question to the LLM ----------
def answer_question(question: str, index: dict, top_k: int = 4):
    client = get_client()
    hits = retrieve(question, index, top_k=top_k)
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,  # low value = more factual answers
    )
    return resp.choices[0].message.content, hits