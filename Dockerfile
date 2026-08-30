FROM python:3.11-slim AS builder
ENV POETRY_VERSION=1.8.4 POETRY_VIRTUALENVS_CREATE=false
RUN pip install --no-cache-dir poetry==$POETRY_VERSION
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-interaction --no-ansi

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app ./app
COPY src ./src
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
