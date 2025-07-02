FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    PATH="${PATH}:/opt/poetry/bin"
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml .
RUN poetry install --no-root --only main

COPY . .

CMD ["poetry", "run", "bash", "run.sh"]