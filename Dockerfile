# Dockerfile — Email Triage Environment
#
# Hugging Face Spaces Docker Spaces requirements:
#   - Must expose port 7860
#   - Must be runnable by a non-root user (uid 1000)
#   - HEALTHCHECK helps HF know when the Space is ready
#
# Local build & run:
#   docker build -t email-triage-env .
#   docker run -p 7860:7860 email-triage-env
#
# With OpenAI key (for /baseline endpoint):
#   docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-triage-env

FROM python:3.11-slim

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────
# Copy only dependency files first so Docker can cache this layer.
# The layer is only invalidated when pyproject.toml changes, not the source.
COPY pyproject.toml .

RUN pip install --no-cache-dir \
    "openenv-core>=0.2.1" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0.0" \
    "websockets>=12.0" \
    "requests>=2.25.0" \
    "openai>=1.0.0"

# ── Copy source ───────────────────────────────────────────────────────────────
# Note: models.py and data.py are included inside the /server/ directory copy
COPY client.py     .
COPY inference.py  .
COPY openenv.yaml  .
COPY server/       ./server/

# Output directories (gitignored at runtime)
RUN mkdir -p outputs/logs outputs/evals \
    && chown -R appuser:appuser /app

USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────
# PORT 7860 is required by Hugging Face Spaces
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

# ── Start server ──────────────────────────────────────────────────────────────
# Runs server.app:main() which calls uvicorn on PORT
CMD ["python", "-c", \
     "import sys; sys.path.insert(0,'/app'); from server.app import main; main()"]
