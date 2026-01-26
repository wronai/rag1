.PHONY: install run index query reindex clean test docker-build docker-run

VENV := .venv
PYTHON := $(shell if [ -x "$(VENV)/bin/python" ]; then echo "$(VENV)/bin/python"; elif [ -x "$(VENV)/Scripts/python.exe" ]; then echo "$(VENV)/Scripts/python.exe"; elif [ -x "$(VENV)/Scripts/python" ]; then echo "$(VENV)/Scripts/python"; else echo "$(VENV)/bin/python"; fi)
FOLDER ?= ./docs
QUERY ?= Co to jest Bolmo?
K ?= 5
BACKEND ?= faiss
CUDA_HOME ?= /usr/local/cuda

install: $(VENV)/.installed

$(VENV)/.installed: install.sh requirements.txt
	./install.sh
	touch $(VENV)/.installed

run: install
	@if [ -n "$(CUDA_HOME)" ] && [ -d "$(CUDA_HOME)" ]; then \
		CUDA_HOME="$(CUDA_HOME)" $(PYTHON) rag.py --backend $(BACKEND) --folder $(FOLDER) --query "$(QUERY)" --k $(K); \
	else \
		$(PYTHON) rag.py --backend $(BACKEND) --folder $(FOLDER) --query "$(QUERY)" --k $(K); \
	fi

index: install
	$(PYTHON) rag.py --backend $(BACKEND) --folder $(FOLDER) --index-only

query: install
	@if [ -n "$(CUDA_HOME)" ] && [ -d "$(CUDA_HOME)" ]; then \
		CUDA_HOME="$(CUDA_HOME)" $(PYTHON) rag.py --backend $(BACKEND) --folder $(FOLDER) --query "$(QUERY)" --k $(K); \
	else \
		$(PYTHON) rag.py --backend $(BACKEND) --folder $(FOLDER) --query "$(QUERY)" --k $(K); \
	fi

reindex: install
	@if [ "$(BACKEND)" = "qdrant" ]; then \
		echo "Starting Qdrant server if not already running..."; \
		docker run -d -p 6333:6333 --name qdrant-server qdrant/qdrant:latest || echo "Qdrant server already running or failed to start"; \
	fi
	BOLMO_FORCE_CPU=1 $(PYTHON) rag.py --backend $(BACKEND) --folder $(FOLDER) --reindex --index-only

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
