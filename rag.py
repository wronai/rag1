import os
import argparse
import json
import csv
from io import StringIO
from typing import TypedDict
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers.configuration_utils as _hf_configuration_utils
import transformers.utils.generic as _hf_generic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


def _ensure_layer_type_validation():
    """Backport transformers.layer_type_validation for models that expect it."""

    if hasattr(_hf_configuration_utils, "layer_type_validation"):
        return

    def layer_type_validation(layer_types):
        if layer_types is None:
            return
        if not isinstance(layer_types, (list, tuple)):
            raise ValueError("layer_types must be a list or tuple of strings")
        for idx, layer_type in enumerate(layer_types):
            if not isinstance(layer_type, str):
                raise ValueError(
                    f"Layer type at position {idx} must be a string, got {type(layer_type)}"
                )

    _hf_configuration_utils.layer_type_validation = layer_type_validation


_ensure_layer_type_validation()


def _ensure_transformers_kwargs():
    if hasattr(_hf_generic, "TransformersKwargs"):
        return

    class TransformersKwargs(TypedDict, total=False):
        pass

    _hf_generic.TransformersKwargs = TransformersKwargs


_ensure_transformers_kwargs()

load_dotenv()


def log(message: str):
    print(f"[rag] {message}", flush=True)

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
    def _read(handle):
        reader = csv.reader(handle)
        lines = []
        for row in reader:
            lines.append(", ".join(row))
        return "\n".join(lines)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return _read(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            buffer = StringIO(f.read())
        return _read(buffer)

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
                log(f"Warning: failed to load {path}: {e}")
                continue

            if text.strip():
                docs.append((path, text))

    return docs

# -----------------------------
# 3) Chunkowanie po akapitach
# -----------------------------

def chunk_text(text, chunk_size=800, overlap=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < chunk_size:
            current += "\n\n" + p
        else:
            chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    # overlap
    if overlap > 0:
        merged = []
        for i, c in enumerate(chunks):
            if i == 0:
                merged.append(c)
            else:
                merged.append(chunks[i-1][-overlap:] + "\n\n" + c)
        return merged

    return chunks

# -----------------------------
# 4) FAISS persistencja
# -----------------------------

def save_faiss(index, path):
    faiss.write_index(index, path)

def load_faiss(path):
    return faiss.read_index(path)

# -----------------------------
# 5) Budowa FAISS index (HNSW)
# -----------------------------

def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode([c for _, c in chunks], convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 40
    index.add(embeddings)

    return index, embeddings

# -----------------------------
# 6) Retrieval
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
# 7) Generowanie odpowiedzi (Bolmo)
# -----------------------------

def generate_answer(question, docs, model, tokenizer, device):
    sources = []
    for idx, (p, c) in enumerate(docs, start=1):
        sources.append(f"[{idx}] {os.path.basename(p)}\n{c}")
    context = "\n\n".join(sources)

    prompt = (
        "Jesteś pomocnym asystentem.\n"
        "Odpowiedz WYŁĄCZNIE na pytanie, bez powtarzania kontekstu ani poleceń.\n"
        "Używaj TYLKO informacji z kontekstu.\n"
        "Jeśli odpowiedzi nie ma w kontekście, napisz dokładnie: Nie mam informacji w dokumentach.\n\n"
        "Kontekst (źródła oznaczone [1], [2], ...):\n"
        f"{context}\n\n"
        f"Pytanie: {question}\n"
        "Odpowiedź (po polsku, markdown, z cytowaniami [1], [2]):\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    try:
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    except TypeError as e:
        log(f"TypeError in generate call: {e}")
        log("Trying alternative generate call signature...")
        try:
            outputs = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        except Exception as e2:
            log(f"Alternative generate call failed: {e2}")
            raise

    sequences = getattr(outputs, "sequences", outputs)
    generated = sequences[0]
    generated_only = generated[input_len:]
    if generated_only.numel() == 0:
        return ""

    text = tokenizer.decode(generated_only, skip_special_tokens=True).strip()

    for marker in ("Odpowiedź", "Answer"):
        marker_colon = marker + ":"
        if marker_colon in text:
            text = text.split(marker_colon, 1)[1].strip()
            break
    for marker in ("Kontekst", "Context", "Pytanie", "Question"):
        marker_colon = marker + ":"
        if marker_colon in text:
            text = text.split(marker_colon, 1)[0].strip()

    return text

# -----------------------------
# 8) Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Bolmo RAG - folder -> query -> markdown output")
    parser.add_argument("--folder", required=True, help="Ścieżka do folderu z dokumentami")
    parser.add_argument("--query", required=True, help="Zapytanie")
    parser.add_argument("--k", type=int, default=5, help="Ile fragmentów pobrać (retrieval)")
    args = parser.parse_args()

    log(f"Ładowanie dokumentów z {args.folder}...")
    docs = load_all_docs(args.folder)
    if not docs:
        log("Brak dokumentów w podanym folderze.")
        return
    log(f"Załadowano {len(docs)} plików.")

    log("Chunkowanie dokumentów...")
    chunks = []
    for path, text in docs:
        for chunk in chunk_text(text):
            chunks.append((path, chunk))
    log(f"Chunkowanie zakończone – {len(chunks)} fragmentów.")

    embedder_name = os.getenv("BOLMO_EMBEDDER", "all-MiniLM-L6-v2")
    log(f"Ładowanie embeddera {embedder_name}...")
    embedder = SentenceTransformer(embedder_name)

    index_path = os.path.join(args.folder, "faiss.index")

    if os.path.exists(index_path):
        log("Ładowanie istniejącego indexu FAISS...")
        index = load_faiss(index_path)
    else:
        log("Budowa indexu FAISS...")
        index, _ = build_faiss_index(chunks, embedder)
        save_faiss(index, index_path)
        log("Index zapisany na dysku.")

    log(f"Retrieval – wyszukiwanie {args.k} fragmentów...")
    retrieved = retrieve(args.query, chunks, index, embedder, k=args.k)
    log(f"Pobrano {len(retrieved)} fragmentów kontekstu.")

    force_cpu = os.getenv("BOLMO_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
    if force_cpu and hasattr(torch, "cuda"):
        torch.cuda.is_available = lambda: False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and "CUDA_HOME" not in os.environ:
        candidates = (
            "/usr/local/cuda",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.3",
            "/usr/local/cuda-12.2",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
        )
        for cand in candidates:
            if os.path.isdir(os.path.join(cand, "include")):
                os.environ["CUDA_HOME"] = cand
                log(f"Ustawiam CUDA_HOME={cand}")
                break

        if "CUDA_HOME" not in os.environ:
            log(
                "CUDA wykryte, ale CUDA_HOME nie jest ustawione i nie udało się go auto-wykryć. "
                "Ustaw CUDA_HOME na katalog instalacji CUDA Toolkit (np. /usr/local/cuda) lub uruchom z BOLMO_FORCE_CPU=1. "
                "Przechodzę na CPU."
            )
            if hasattr(torch, "cuda"):
                torch.cuda.is_available = lambda: False
            device = "cpu"

    model_name = os.getenv("BOLMO_MODEL", "allenai/Bolmo-1B")
    log(f"Ładowanie modelu {model_name} na urządzeniu {device}...")

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as err:
        log(
            "Nie udało się załadować szybkiego tokenizera (use_fast=True) – "
            "przechodzę na wersję wolną."
        )
        log(f"Szczegóły: {err}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )
        except Exception:
            raise

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    except Exception as e:
        log(f"Error loading model with default settings: {e}")
        log("Attempting to load model with additional compatibility settings...")
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if hasattr(config, 'rope_scaling') and hasattr(config.rope_scaling, 'type'):
                config.rope_scaling.type = 'linear'  # Use a known valid type
            elif hasattr(config, 'rope_type'):
                config.rope_type = 'linear'  # Fallback to a known type if 'default' is not supported
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True, ignore_mismatched_sizes=True).to(device)
        except Exception as e2:
            log(f"Failed to load model with compatibility settings: {e2}")
            raise

    log("Generowanie odpowiedzi...")
    answer = generate_answer(args.query, retrieved, model, tokenizer, device)

    print(answer)


if __name__ == "__main__":
    main()
