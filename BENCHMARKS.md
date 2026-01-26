# Benchmarki (3 warianty RAG)

Poniżej są 3 warianty retrievalu, które możesz uruchomić na tym samym korpusie `./docs`:

- `faiss_nocache` – baseline: za każdym razem load + chunk + embed + budowa FAISS
- `faiss` – FAISS z cache w `./docs/.rag_cache` (szybkie kolejne uruchomienia)
- `qdrant` – Qdrant (vectordb) + ingest, potem szybkie wyszukiwanie

## 1) FAISS bez cache (baseline)

```bash
make run BACKEND=faiss_nocache FOLDER=./docs QUERY="Co to jest Bolmo?" K=5
```

## 2) FAISS z cache (zalecane lokalnie)

Pierwszy raz (zbuduje cache):

```bash
make index BACKEND=faiss FOLDER=./docs
```

Potem szybkie query:

```bash
make run BACKEND=faiss FOLDER=./docs QUERY="Co to jest Bolmo?" K=5
```

Wymuszenie przebudowy cache:

```bash
make reindex BACKEND=faiss FOLDER=./docs
```

## 3) Qdrant

Uruchom Qdrant lokalnie:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:latest
```

Ingest (indexowanie do Qdrant):

```bash
make index BACKEND=qdrant FOLDER=./docs
```

Query:

```bash
make run BACKEND=qdrant FOLDER=./docs QUERY="Co to jest Bolmo?" K=5
```

## Benchmark (tabela: szybkość + hit@k)

Skrypt:

```bash
python benchmarks/benchmark_rag.py --folder ./docs --k 5 --reindex
```

Wynik to tabela markdown z polami:

- `index_s` – czas przygotowania (budowy indexu / cache / ingestu)
- `avg_query_ms` – średni czas pojedynczego retrievalu
- `hit@k` – odsetek zapytań, dla których oczekiwany plik pojawił się w top-k
