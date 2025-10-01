FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Upgrade pip, setuptools, wheel and install requirements with extended timeout
# Using pypi.org mirror explicitly for stability
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --default-timeout=100 \
       -i https://pypi.org/simple -r requirements.txt

# Copy the rest of the project
COPY summarizer ./summarizer
COPY download_models.py ./download_models.py
COPY README.md ./README.md

ENTRYPOINT ["python", "-m", "summarizer.summarize"]
CMD ["--help"]
