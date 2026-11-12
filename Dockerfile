FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY etl ./etl
COPY ml ./ml
COPY sql/*.sql ./sql/
COPY tests ./tests

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "etl.pipeline"]
