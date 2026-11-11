# Multi-stage Dockerfile for EchoPulse AI
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for audio and portaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definitions and install
COPY requirements.txt pyproject.toml setup.py ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Generate sample files
RUN python samples/generate_samples.py

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: launch FastAPI server (or override with streamlit)
CMD ["uvicorn", "audio_pulse.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
