#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_EXE="/c/Users/Asus/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "No se encontro el Python con dependencias en:"
  echo "  $PYTHON_EXE"
  echo
  echo "Instala dependencias en otro Python o edita PYTHON_EXE dentro de run_mingw64.sh."
  exit 1
fi

"$PYTHON_EXE" scripts/run_all_local.py "$@"
