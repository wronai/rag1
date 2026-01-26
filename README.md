# Bolmo RAG Toolkit

Zestaw narzędzi do eksperymentów z Retrieval-Augmented Generation (RAG) oparty na skrypcie `rag.py`. Repozytorium zawiera gotowy skrypt instalacyjny `install.sh` oraz dopasowany `Makefile`, które w kilka sekund przygotowują środowisko uruchomieniowe i dostarczają wygodne cele (`install`, `run`, `clean`).

## Wymagania wstępne

- Linux / macOS z `python3` i `venv`
- Dostęp do internetu w celu pobrania zależności Pythona

## Instalacja środowiska (install.sh)

1. Nadaj plikowi uprawnienia wykonywalne:

   ```bash
   chmod +x install.sh
   ```

2. Uruchom skrypt:

   ```bash
   ./install.sh
   ```

Skrypt:

- tworzy virtualenv (`.venv`),
- aktywuje środowisko,
- aktualizuje `pip`,
- instaluje komplet zależności zpinowanych pod Python 3.13/3.11, aby uniknąć kompilacji Rust/C (`transformers==4.46.2`, `tokenizers==0.20.3`, `sentence-transformers==2.6.1`, `faiss-cpu==1.13.2`, `torch==2.10.0`, `PyPDF2==3.0.1`, `python-docx==1.2.0`, `html2text==2025.4.15`, `python-dotenv==1.0.1`).

Po zakończeniu aktywuj środowisko komendą `source .venv/bin/activate`.

> **Uwaga:** jeśli pracujesz na starszej dystrybucji, upewnij się, że masz zainstalowane `build-essential`, `rustc` oraz `cargo`; nowe paczki powinny jednak instalować się z gotowych kółek (wheels) i nie wymagać kompilacji.


## Przykładowa kolekcja `docs/`

Repozytorium zawiera katalog `docs/` z wieloma formatami, które możesz wykorzystać do szybkiego testu pipeline'u:

- `readme.txt` – tekstowe wprowadzenie do korpusu.
- `faq.md` – markdown z najczęstszymi pytaniami.
- `support.html` – prosty dokument HTML do sprawdzenia ekstrakcji z treści webowych.
- `schedule.csv` – tabela z harmonogramem zadań.
- `config.json` – przykładowa konfiguracja indeksowania.
- `meeting_notes.docx` – notatki ze spotkania w formacie Word.
- `overview.pdf` – krótki PDF opisujący demo Bolmo RAG.

Aby uruchomić demo na tych danych:

```bash
make run FOLDER=./docs QUERY="Co zawiera pakiet demo?"
```

### Konfiguracja modeli przez `.env`

1. Sklonuj plik wzorcowy i ustaw zmienną:

   ```bash
   cp .env.example .env
   ```

2. Otwórz `.env` i ustaw:
   - `BOLMO_MODEL` – model generatywny (domyślnie `allenai/Bolmo-1B`). Jeśli masz zasoby GPU, możesz przełączyć na `allenai/Bolmo-7B`.
   - `BOLMO_EMBEDDER` – sentence-transformer do częśći retrieval (domyślnie `all-MiniLM-L6-v2`). Możesz wskazać np. `sentence-transformers/all-mpnet-base-v2`.

Skrypt `rag.py` ładuje `.env` automatycznie dzięki `python-dotenv`, więc każda zmiana wartości jest widoczna przy kolejnym uruchomieniu `make run`.

## Automatyzacja za pomocą Makefile

Makefile udostępnia zestaw celów:

```bash
make install                      # tworzy .venv poprzez install.sh
make run FOLDER=./docs QUERY="?"   # uruchamia rag.py z parametrami
make test                         # uruchamia pytest
make clean                        # usuwa katalog .venv
make docker-build                 # buduje obraz Dockera bolmo-rag
make docker-run FOLDER=... QUERY=... # buduje i uruchamia obraz z zamontowanymi docs/.env
make docker-shell                 # wchodzi do python:3.11 z repo pod /app
make docker-up / docker-logs / docker-stop # zarządza kontenerem w tle
```

### Opis celów

- **install** – zależy od pliku `.venv/bin/activate`; jeżeli środowisko nie istnieje, uruchamia `install.sh` i tworzy plik znacznika w katalogu `.venv`.
- **run** – zapewnia, że środowisko jest gotowe (wywołuje `make install`), a następnie uruchamia `rag.py` z przekazanymi argumentami `--folder` oraz `--query`.
- **test** – uruchamia `pytest`, aby sprawdzić podstawowe wczytywanie dokumentów.
- **clean** – usuwa katalog `.venv`, pozwalając rozpocząć instalację od zera.
- **docker-build** – pakuje aplikację w obraz Dockera z wykorzystaniem `requirements.txt`.
- **docker-run** – uruchamia wcześniej zbudowany obraz z podmontowanym katalogiem `docs/` i plikiem `.env`, dzięki czemu można zmieniać korpusy bez przebudowy obrazu.
- **docker-shell** – otwiera tymczasowy kontener `python:3.11` z repozytorium zamontowanym pod `/app`; idealne do ręcznego testowania `pip install -r requirements.txt && make run ...` w kontrolowanym środowisku.
- **docker-up** – startuje kontener w tle (nazwa `bolmo-rag-run`),
- **docker-logs** – tailuje logi działającego kontenera,
- **docker-stop** – zatrzymuje i usuwa kontener uruchomiony przez `docker-up`.

### Scenariusz „Docker + logi”

1. Zbuduj obraz i odpal pipeline w tle:

   ```bash
   make docker-up FOLDER=./docs QUERY="Co to jest Bolmo?"
   ```

2. Podejrzyj logi na żywo (działa jak `docker logs -f`):

   ```bash
   make docker-logs
   ```

3. Po zakończeniu zatrzymaj kontener:

   ```bash
   make docker-stop
   ```

Jeżeli chcesz jedynie wejść w interaktywnego `bash`a w obrazie referencyjnym, uruchom `make docker-shell`, a następnie w kontenerze wykonaj `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && make run ...`.

### Przykładowe użycie celu `run`

```bash
make run FOLDER=./docs QUERY="Co to jest Bolmo?"
```

Zmodyfikuj wartości `FOLDER` i `QUERY`, aby wskazać własny zbiór dokumentów oraz pytanie kierowane do modułu RAG.

### Reset i ponowna instalacja środowiska

Jeśli chcesz upewnić się, że pracujesz na świeżej konfiguracji, wykonaj kolejno:

```bash
make clean                              # usuwa poprzednią virtualenv
make install                            # ponownie tworzy środowisko i instaluje zależności
make run FOLDER=./docs QUERY="Co to jest Bolmo?"  # uruchamia RAG na próbkach z docs/
```

Pierwsze dwa kroki gwarantują czyste środowisko, a trzeci demonstruje pełne wywołanie pipeline'u na przykładowej kolekcji.


## Dalsze kroki

- Rozbuduj `rag.py` (lub alternatywny skrypt, np. `bolmo_rag.py`) o własne logiki indeksowania czy interfejs CLI. Makefile i struktura środowiska są gotowe na kolejne rozszerzenia.

Powodzenia w pracy z Bolmo RAG!
