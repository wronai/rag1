import argparse
import os
import time

import rag


def _now():
    return time.perf_counter()


def _hit_at_k(retrieved, expected_name: str) -> bool:
    expected_name = expected_name.lower()
    for p, _ in retrieved:
        if os.path.basename(p).lower() == expected_name:
            return True
    return False


def _bench_faiss_nocache(folder: str, embedder, chunk_size: int, overlap: int, queries, k: int):
    t0 = _now()
    docs = rag.load_all_docs(folder)
    chunks = []
    for path, text in docs:
        for ch in rag.chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append((path, ch))
    index, _ = rag.build_faiss_index(chunks, embedder)
    t1 = _now()

    q_times = []
    hits = 0
    for q, expected in queries:
        qs = _now()
        retrieved = rag.retrieve(q, chunks, index, embedder, k=k)
        qe = _now()
        q_times.append(qe - qs)
        hits += 1 if _hit_at_k(retrieved, expected) else 0

    return {
        "index_s": t1 - t0,
        "avg_query_ms": (sum(q_times) / max(1, len(q_times))) * 1000.0,
        "hit_at_k": hits / max(1, len(queries)),
    }


def _bench_faiss_cached(folder: str, embedder, embedder_name: str, chunk_size: int, overlap: int, queries, k: int, reindex: bool):
    t0 = _now()
    chunks, index = rag._ensure_faiss_cache(
        folder=folder,
        embedder=embedder,
        embedder_name=embedder_name,
        reindex=reindex,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    t1 = _now()

    q_times = []
    hits = 0
    for q, expected in queries:
        qs = _now()
        retrieved = rag.retrieve(q, chunks, index, embedder, k=k)
        qe = _now()
        q_times.append(qe - qs)
        hits += 1 if _hit_at_k(retrieved, expected) else 0

    return {
        "index_s": t1 - t0,
        "avg_query_ms": (sum(q_times) / max(1, len(q_times))) * 1000.0,
        "hit_at_k": hits / max(1, len(queries)),
    }


def _bench_qdrant(folder: str, embedder, chunk_size: int, overlap: int, queries, k: int, reindex: bool):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "bolmo_docs")

    client = QdrantClient(url=qdrant_url)
    exists = client.collection_exists(collection_name=collection)

    t0 = _now()
    if reindex and exists:
        client.delete_collection(collection_name=collection)
        exists = False

    if reindex or not exists:
        docs = rag.load_all_docs(folder)
        chunks = []
        for path, text in docs:
            for ch in rag.chunk_text(text, chunk_size=chunk_size, overlap=overlap):
                chunks.append((path, ch))
        vecs = embedder.encode([c for _, c in chunks], convert_to_numpy=True)
        dim = int(vecs.shape[1])

        if not exists:
            client.create_collection(
                collection_name=collection,
                vectors_config=qdrant_models.VectorParams(size=dim, distance=qdrant_models.Distance.COSINE),
            )

        points = []
        for (p, c), v in zip(chunks, vecs):
            points.append(qdrant_models.PointStruct(id=os.path.basename(p) + "_" + str(abs(hash(c))), vector=v.tolist(), payload={"path": p, "text": c}))

        batch = 128
        for i in range(0, len(points), batch):
            client.upsert(collection_name=collection, points=points[i : i + batch])
    t1 = _now()

    q_times = []
    hits = 0
    for q, expected in queries:
        vec = embedder.encode([q], convert_to_numpy=True)[0].tolist()
        qs = _now()
        res = client.search(collection_name=collection, query_vector=vec, limit=k, with_payload=True)
        qe = _now()
        q_times.append(qe - qs)
        retrieved = [(r.payload.get("path", ""), r.payload.get("text", "")) for r in res]
        hits += 1 if _hit_at_k(retrieved, expected) else 0

    return {
        "index_s": t1 - t0,
        "avg_query_ms": (sum(q_times) / max(1, len(q_times))) * 1000.0,
        "hit_at_k": hits / max(1, len(queries)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./docs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--embedder", default=os.getenv("BOLMO_EMBEDDER", "all-MiniLM-L6-v2"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("BOLMO_CHUNK_SIZE", "800")))
    parser.add_argument("--overlap", type=int, default=int(os.getenv("BOLMO_CHUNK_OVERLAP", "200")))
    args = parser.parse_args()

    embedder = rag.SentenceTransformer(args.embedder)

    queries = [
        ("Czym jest Bolmo?", "faq.md"),
        ("Kontakt z zespołem Bolmo", "support.html"),
        ("Harmonogram wdrożeń", "meeting_notes.docx"),
    ]

    results = []
    results.append(("faiss_nocache", _bench_faiss_nocache(args.folder, embedder, args.chunk_size, args.overlap, queries, args.k)))
    results.append(("faiss_cached", _bench_faiss_cached(args.folder, embedder, args.embedder, args.chunk_size, args.overlap, queries, args.k, args.reindex)))

    try:
        results.append(("qdrant", _bench_qdrant(args.folder, embedder, args.chunk_size, args.overlap, queries, args.k, args.reindex)))
    except Exception as e:
        results.append(("qdrant", {"error": str(e)}))

    print("| backend | index_s | avg_query_ms | hit@k |")
    print("|---|---:|---:|---:|")
    for name, r in results:
        if "error" in r:
            print(f"| {name} | - | - | - |")
            continue
        print(f"| {name} | {r['index_s']:.3f} | {r['avg_query_ms']:.2f} | {r['hit_at_k']:.2f} |")


if __name__ == "__main__":
    main()
