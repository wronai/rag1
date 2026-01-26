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
- instaluje komplet zależności: `transformers==4.34.0`, `sentence-transformers`, `faiss-cpu`, `torch`, `PyPDF2`, `python-docx`, `html2text`, `pandas`.

Po zakończeniu aktywuj środowisko komendą `source .venv/bin/activate`.

## Automatyzacja za pomocą Makefile

Makefile udostępnia trzy główne cele:

```bash
make install                 # tworzy .venv poprzez install.sh
make run FOLDER=./ QUERY="Co to jest Bolmo?"   # uruchamia rag.py z parametrami
make clean                   # usuwa katalog .venv
```

### Opis celów

- **install** – zależy od pliku `.venv/bin/activate`; jeżeli środowisko nie istnieje, uruchamia `install.sh` i tworzy plik znacznika w katalogu `.venv`.
- **run** – zapewnia, że środowisko jest gotowe (wywołuje `make install`), a następnie uruchamia `rag.py` z przekazanymi argumentami `--folder` oraz `--query`.
- **clean** – usuwa katalog `.venv`, pozwalając rozpocząć instalację od zera.

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

## Dalsze kroki

- Dodaj dodatkowe cele (`test`, `lint`, generowanie raportów) w Makefile, jeżeli potrzebujesz bardziej rozbudowanych przepływów pracy.
- Rozbuduj `rag.py` (lub alternatywny skrypt, np. `bolmo_rag.py`) o własne logiki indeksowania czy interfejs CLI. Makefile i struktura środowiska są gotowe na kolejne rozszerzenia.

Powodzenia w pracy z Bolmo RAG!
