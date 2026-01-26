.PHONY: install run clean

VENV := .venv
PYTHON := $(VENV)/bin/python
FOLDER ?= ./docs
QUERY ?= Co to jest Bolmo?
K ?= 5

install: $(VENV)/bin/activate

$(VENV)/bin/activate: install.sh
	./install.sh
	touch $(VENV)/bin/activate

run: install
	$(PYTHON) rag.py --folder $(FOLDER) --query "$(QUERY)" --k $(K)

clean:
	rm -rf .venv

help:
	@echo "make install  # tworzy/aktualizuje .venv"
	@echo "make run      # uruchamia rag.py (domyślnie folder $(FOLDER), zapytanie '$(QUERY)', k=$(K))"
	@echo "make clean    # usuwa .venv"
	@echo "Zmienne: FOLDER=/ścieżka QUERY=\"tekst\" K=liczba"
