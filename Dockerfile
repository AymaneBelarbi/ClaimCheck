FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so this layer is cached until the lockfile changes. No dependency groups:
# the dataset generator and the dev tools stay on the host.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY . .
RUN uv sync --frozen --no-default-groups

# The services run as an unprivileged user. The two directories that compose mounts volumes
# on are created here and owned by it, so a new volume starts out writable.
RUN useradd --system --uid 10001 --home-dir /app --no-create-home claimcheck \
    && mkdir -p /data/storage /app/var/models \
    && chown -R claimcheck /data/storage /app/var
USER claimcheck

EXPOSE 8000
CMD ["uvicorn", "claimcheck.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
