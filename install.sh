#!/usr/bin/env bash
set -e

echo "=== Tworzenie virtualenv ==="
python3 -m venv .venv

echo "=== Aktywacja virtualenv ==="
source .venv/bin/activate

echo "=== Aktualizacja pip ==="
pip install --upgrade pip

echo "=== Instalacja zależności ==="
pip install \
  transformers==4.39.0 \
  sentence-transformers \
  faiss-cpu \
  torch \
  PyPDF2 \
  python-docx \
  html2text \
  pandas

echo "=== Gotowe! ==="
echo "Aby aktywować środowisko: source .venv/bin/activate"
