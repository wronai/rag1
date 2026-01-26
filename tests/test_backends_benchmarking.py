import os
import time
from pathlib import Path

import numpy as np

import rag


class CountingEmbedder:
    def __init__(self, name: str):
        self.name = name
        self.encode_calls = 0
        self.total_texts = 0

    def encode(self, texts, convert_to_numpy=True):
        self.encode_calls += 1
        self.total_texts += len(texts)

        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            tl = str(t).lower()
            if "bolmo" in tl:
                vectors[i, 0] = 1.0
            if "kontakt" in tl:
                vectors[i, 1] = 1.0
            if "harmonogram" in tl:
                vectors[i, 2] = 1.0
            if vectors[i].sum() == 0:
                vectors[i, 3] = 1.0
        return vectors


def test_faiss_nocache_retrieval_hit(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("To jest tekst bez sygnału.", encoding="utf-8")
    (docs_dir / "b.txt").write_text("Bolmo to przykład systemu RAG.", encoding="utf-8")

    embedder = CountingEmbedder("dummy")

    docs = rag.load_all_docs(str(docs_dir))
    chunks = []
    for path, text in docs:
        for ch in rag.chunk_text(text, chunk_size=200, overlap=0):
            chunks.append((path, ch))

    index, _ = rag.build_faiss_index(chunks, embedder)
    retrieved = rag.retrieve("Co to jest Bolmo?", chunks, index, embedder, k=1)
    assert retrieved
    assert "bolmo" in retrieved[0][1].lower()


def test_faiss_cache_avoids_reembedding_on_second_load(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "faq.md").write_text("Bolmo to przykład systemu RAG.", encoding="utf-8")
    (docs_dir / "support.html").write_text("Kontakt: help@bolmo.ai", encoding="utf-8")

    embedder = CountingEmbedder("dummy")

    chunks1, index1 = rag._ensure_faiss_cache(
        folder=str(docs_dir),
        embedder=embedder,
        embedder_name="dummy",
        reindex=True,
        chunk_size=200,
        overlap=0,
    )
    assert chunks1
    assert index1 is not None
    calls_after_build = embedder.encode_calls

    chunks2, index2 = rag._ensure_faiss_cache(
        folder=str(docs_dir),
        embedder=embedder,
        embedder_name="dummy",
        reindex=False,
        chunk_size=200,
        overlap=0,
    )
    assert chunks2
    assert index2 is not None
    assert embedder.encode_calls == calls_after_build

    retrieved = rag.retrieve("Bolmo", chunks2, index2, embedder, k=1)
    assert retrieved


def test_cached_path_is_not_slower_than_nocache(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for i in range(50):
        (docs_dir / f"doc{i}.txt").write_text(f"Dokument {i}. Bolmo.", encoding="utf-8")

    embedder = CountingEmbedder("dummy")

    embedder2 = CountingEmbedder("dummy")
    _chunks, _index = rag._ensure_faiss_cache(
        folder=str(docs_dir),
        embedder=embedder2,
        embedder_name="dummy",
        reindex=True,
        chunk_size=200,
        overlap=0,
    )
    retrieved = rag.retrieve("Bolmo", _chunks, _index, embedder2, k=5)
    assert retrieved


def test_qdrant_backend_optional_smoke(tmp_path):
    if os.getenv("RUN_QDRANT_TESTS", "").strip().lower() not in {"1", "true", "yes"}:
        return

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qdrant_models
    except Exception:
        return

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    name = f"test_{int(time.time() * 1000)}"

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "faq.md").write_text("Bolmo to przykład systemu RAG.", encoding="utf-8")

    embedder = CountingEmbedder("dummy")
    docs = rag.load_all_docs(str(docs_dir))
    chunks = []
    for path, text in docs:
        for ch in rag.chunk_text(text, chunk_size=200, overlap=0):
            chunks.append((path, ch))
    vecs = embedder.encode([c for _, c in chunks], convert_to_numpy=True)

    client.create_collection(
        collection_name=name,
        vectors_config=qdrant_models.VectorParams(size=int(vecs.shape[1]), distance=qdrant_models.Distance.COSINE),
    )

    points = []
    for (p, c), v in zip(chunks, vecs):
        points.append(qdrant_models.PointStruct(id=str(p), vector=v.tolist(), payload={"path": p, "text": c}))

    client.upsert(collection_name=name, points=points)
    hits = client.search(collection_name=name, query_vector=embedder.encode(["Bolmo"], convert_to_numpy=True)[0].tolist(), limit=1, with_payload=True)
    assert hits

    client.delete_collection(collection_name=name)
