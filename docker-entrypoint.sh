#!/bin/bash
# Entrypoint script for BabbageBox Backend
# Provides better debugging and graceful startup

set -e

echo "========================================="
echo "BabbageBox Backend Starting"
echo "========================================="
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "MODELS_DIR: ${MODELS_DIR:-/models}"
echo "LLAMA_MODEL_PATH: ${LLAMA_MODEL_PATH:-auto-detect}"
echo "========================================="

# List models directory
if [ -d "${MODELS_DIR:-/models}" ]; then
    echo "Models found:"
    ls -la "${MODELS_DIR:-/models}" || echo "No models directory or empty"
else
    echo "WARNING: Models directory does not exist: ${MODELS_DIR:-/models}"
fi

# Check if we can import the app
echo "Testing Python imports..."
python -c "
import sys
sys.stdout.flush()
print('Python imports starting...')
try:
    print('Importing FastAPI...')
    from fastapi import FastAPI
    print('Importing main app...')
    from app.main import app
    print('All imports successful!')
except Exception as e:
    print(f'Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()
"

echo "========================================="
echo "Starting uvicorn..."
echo "========================================="

# Run with unbuffered output
exec python -u -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level debug
