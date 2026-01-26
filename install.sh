#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
SENTINEL_FILE="$VENV_DIR/.installed"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "[install] ERROR: Nie znaleziono interpretera Pythona (python3/python)." >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("[install] ERROR: Wymagany Python >= 3.11 (dla xlstm).")
PY

echo "=== Tworzenie virtualenv ==="
"$PYTHON" -m venv "$VENV_DIR"

echo "=== Aktywacja virtualenv ==="
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  source "$VENV_DIR/Scripts/activate"
else
  echo "[install] ERROR: Nie mogę znaleźć skryptu aktywacji virtualenv." >&2
  exit 1
fi

echo "=== Aktualizacja pip ==="
python -m pip install --upgrade pip

echo "=== Instalacja zależności ==="
python -m pip install -r requirements.txt

mkdir -p "$VENV_DIR"
printf "installed\n" > "$SENTINEL_FILE"

echo "=== Gotowe! ==="
echo "Aby aktywować środowisko (Unix/WSL): source .venv/bin/activate"
echo "Aby aktywować środowisko (Windows Git-Bash): source .venv/Scripts/activate"
