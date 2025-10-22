# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

# Install python deps first to leverage caching
COPY requirements.txt requirements.txt
COPY tradingbot_ibkr/requirements.txt tradingbot_ibkr/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r tradingbot_ibkr/requirements.txt

# Copy application source
COPY . .

EXPOSE 8000

ENV MCP_BASE_URL="" \
    MCP_API_KEY="" \
    SECRET_KEY="change-this"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
