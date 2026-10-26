#!/bin/bash
# Run from the extracted project folder; first run needs internet for packages.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import flask, cv2, numpy' 2>/dev/null; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python download_model.py
.venv/bin/python app.py
