# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage build. The builder installs the exact dependency set pinned in
# uv.lock into a virtualenv; the runtime stage copies only that virtualenv and
# the installed package, so the shipped image carries no compilers, no uv, no
# lockfile and no build metadata.
#
# UV_EXTRAS selects optional dependency groups, so one Dockerfile produces both
# the lean API image and the dashboard image:
#   docker build --target runtime .
#   docker build --target runtime --build-arg UV_EXTRAS="--extra dashboard" .
# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS builder

ARG UV_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

WORKDIR /build

# Dependency layer: rebuilt only when the lockfile or manifest changes.
# --no-install-project keeps the source out of this layer so editing src/ does
# not invalidate the (slow) dependency install.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable ${UV_EXTRAS}

# Source layer.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable ${UV_EXTRAS}


FROM python:3.12-slim-bookworm AS runtime

# The environment is deliberately not pinned to `production` here: that mode
# rejects the default mock VLM backend, so an image built with it could not
# start without credentials. Deployments set SHELF_ENVIRONMENT=production
# alongside the API key that makes it valid.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SHELF_LOG_JSON=true

# opencv-python-headless links no GUI libraries, so libGL and libxcb are not
# needed here; a `[tool.uv] override-dependencies` entry keeps ultralytics from
# pulling in the GUI build that would need them. libgomp1 is torch's OpenMP
# runtime, and curl backs the healthcheck below.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 shelf \
    && useradd --system --uid 1001 --gid shelf --create-home shelf

COPY --from=builder --chown=shelf:shelf /opt/venv /opt/venv

WORKDIR /app
COPY --chown=shelf:shelf data/sample_data ./data/sample_data

USER shelf
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "smart_shelf.api.app:create_app", "--factory"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
