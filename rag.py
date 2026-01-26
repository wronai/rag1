import os
import argparse
import json
import pandas as pd
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# PDF
from PyPDF2 import PdfReader

# DOCX
import docx

# HTML
import html2text

# -----------------------------
# 1) Loadery dokumentów
# -----------------------------

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_md(path):
    return load_txt(path)

def load_pdf(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    return html2text.html2text(html)

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8", errors="ignore")
    return df.to_string()

def load_json(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False)

# -----------------------------
# 2) Zbieranie dokumentów
# -----------------------------

def load_all_docs(folder):
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            try:
                if ext == ".txt":
                    text = load_txt(path)
                elif ext == ".md":
                    text = load_md(path)
                elif ext == ".pdf":
                    text = load_pdf(path)
                elif ext == ".docx":
                    text = load_docx(path)
                elif ext == ".html" or ext == ".htm":
                    text = load_html(path)
                elif ext == ".csv":
                    text = load_csv(path)
                elif ext == ".json":
                    text = load_json(path)
                else:
                    continue
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
                continue

            if text.strip():
                docs.append((path, text))

    return docs

# -----------------------------
# 3) Chunkowanie (ważne w RAG)
# -----------------------------

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# -----------------------------
# 4) Budowa FAISS index
# -----------------------------

def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode([c for _, c in chunks], convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# -----------------------------
# 5) Retrieval
# -----------------------------

def retrieve(query, chunks, index, embedder, k=5):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    results = []
    for idx in I[0]:
        path, chunk = chunks[idx]
        results.append((path, chunk))
    return results

# -----------------------------
# 6) Generowanie odpowiedzi (Bolmo)
# -----------------------------

def generate_answer(question, docs, model, tokenizer, device):
    context = "\n".join([f"### {os.path.basename(p)}\n{c}" for p, c in docs])

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question: {question}

Answer in Polish in markdown format:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# 7) Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Bolmo RAG - folder -> query -> markdown output")
    parser.add_argument("--folder", required=True, help="Ścieżka do folderu z dokumentami")
    parser.add_argument("--query", required=True, help="Zapytanie")
    parser.add_argument("--k", type=int, default=5, help="Ile fragmentów pobrać (retrieval)")
    args = parser.parse_args()

    # Load docs
    docs = load_all_docs(args.folder)
    if not docs:
        print("No documents found.")
        return

    # Chunk docs
    chunks = []
    for path, text in docs:
        for chunk in chunk_text(text):
            chunks.append((path, chunk))

    # Load embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Build FAISS
    index, _ = build_faiss_index(chunks, embedder)

    # Retrieve
    retrieved = retrieve(args.query, chunks, index, embedder, k=args.k)

    # Load Bolmo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "allenai/Bolmo-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    # Generate answer
    answer = generate_answer(args.query, retrieved, model, tokenizer, device)

    print(answer)


if __name__ == "__main__":
    main()
