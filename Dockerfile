FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set up app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 7777

# Start Ollama + pull model + start JARVIS server
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
