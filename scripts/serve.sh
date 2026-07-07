#!/usr/bin/env bash
# Run the API (port 8000) and the static web UI (port 8080) together.
set -e

cleanup() { kill 0; }
trap cleanup EXIT

echo "→ API   http://localhost:8000  (docs at /docs)"
uvicorn oncotwin.api.main:app --port 8000 &

echo "→ Web   http://localhost:8080"
python -m http.server 8080 &

wait
