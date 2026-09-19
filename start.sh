#!/usr/bin/env bash
#
# Starts the backend (FastAPI/uvicorn) and frontend (Vite) in parallel.
# Press Ctrl+C to stop both.
#
# Usage:
#   ./start.sh
#

set -euo pipefail

# Resolve the directory this script lives in, so it works from anywhere.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}

# Run cleanup on Ctrl+C / termination.
trap cleanup INT TERM EXIT

echo "Starting backend (uvicorn) on http://localhost:8000 ..."
(
  cd "$ROOT_DIR"
  uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
) &
BACKEND_PID=$!

echo "Starting frontend (vite) on http://localhost:5173 ..."
(
  cd "$ROOT_DIR/dashboard"
  npm run dev
) &
FRONTEND_PID=$!

echo ""
echo "Backend  PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop both."
echo ""

# Wait for either process to exit; if one dies, cleanup tears down the other.
wait -n 2>/dev/null || wait
