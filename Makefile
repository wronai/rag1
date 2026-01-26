.PHONY: install run clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install: $(VENV)/bin/activate

$(VENV)/bin/activate: install.sh
	./install.sh
	touch $(VENV)/bin/activate

run: install
	$(PYTHON) rag.py --folder $(FOLDER) --query "$(QUERY)"

clean:
	rm -rf .venv
