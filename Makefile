.PHONY: install run clean test docker-build docker-run

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

test: install
	$(PYTHON) -m pytest -q

clean:
	rm -rf .venv

docker-build:
	docker build -t bolmo-rag:latest .

docker-run: docker-build
	docker run --rm -it \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/.env:/app/.env:ro \
		bolmo-rag:latest \
		--folder $(FOLDER) --query "$(QUERY)" --k $(K)

help:
	@echo "make install  # tworzy/aktualizuje .venv"
	@echo "make run      # uruchamia rag.py (domyślnie folder $(FOLDER), zapytanie '$(QUERY)', k=$(K))"
	@echo "make clean    # usuwa .venv"
	@echo "make test     # uruchamia pytest"
	@echo "make docker-build # buduje obraz Dockera"
	@echo "make docker-run   # buduje i uruchamia obraz z zamontowanymi docs/.env"
	@echo "Zmienne: FOLDER=/ścieżka QUERY=\"tekst\" K=liczba"
