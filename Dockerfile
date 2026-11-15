FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md SPEC.md ./
COPY src ./src
COPY tests ./tests
COPY .streamlit ./.streamlit

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
