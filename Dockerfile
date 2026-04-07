# Dockerfile — Email Triage Environment
#
# Hugging Face Spaces Docker requirements:
#   - Must expose port 7860
#   - Must run as non-root user (uid 1000)
#
# Local build & run:
#   docker build -t email-triage-env .
#   docker run -p 7860:7860 email-triage-env

FROM python:3.11-slim

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────
COPY pyproject.toml .

RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.0" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0.0" \
    "websockets>=12.0" \
    "requests>=2.25.0" \
    "openai>=1.0.0"

# ── Copy source ───────────────────────────────────────────────────────────────
COPY client.py     .
COPY inference.py  .
COPY openenv.yaml  .
COPY server/       ./server/

# Output directories
RUN mkdir -p outputs/logs outputs/evals \
    && chown -R appuser:appuser /app

USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
