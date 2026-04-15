#!/bin/bash
set -e

echo "=== J.A.R.V.I.S. Railway Deployment ==="

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

# Pull gemma4 cloud model (this registers the cloud model)
echo "Registering gemma4:31b-cloud..."
ollama pull gemma4:31b-cloud 2>/dev/null || echo "Cloud model registered"

# Start JARVIS server
echo "Starting JARVIS server on port ${PORT:-7777}..."
exec python server.py
