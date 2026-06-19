#!/bin/bash
set -e

echo "=== J.A.R.V.I.S. Railway Deployment ==="
OLLAMA_MODEL="${OLLAMA_MODEL:-kimi-k2.7-code:cloud}"
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# Start Ollama in background
echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready!"
        break
    fi
    sleep 1
done

# Pull/register the configured Ollama model. Cloud models stay remote, but this
# makes the local Ollama server aware of the model name before JARVIS starts.
echo "Registering ${OLLAMA_MODEL}..."
ollama pull "${OLLAMA_MODEL}" 2>/dev/null || echo "Model registration skipped or already available"

# Start JARVIS server
echo "Starting JARVIS server on port ${PORT:-7777}..."
exec python server.py
