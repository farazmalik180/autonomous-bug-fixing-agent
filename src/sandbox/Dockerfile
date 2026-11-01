FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies are baked in: containers run with the network disabled.
RUN pip install --no-cache-dir "pytest>=8.0"

# Non-root user. uid must match SandboxConfig.uid.
RUN useradd --create-home --uid 1000 --shell /bin/bash sandbox \
    && mkdir -p /workspace \
    && chown sandbox:sandbox /workspace

WORKDIR /workspace
USER sandbox

CMD ["sleep", "infinity"]
