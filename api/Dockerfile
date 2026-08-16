FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY uv.lock pyproject.toml ./


RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY api/ ./api/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]