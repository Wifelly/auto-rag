FROM python:3.12-slim

ENV POETRY_HOME="/opt/poetry" \
    POETRY_URL="https://install.python-poetry.org" \
    POETRY_VERSION=2.0.1

WORKDIR /app

RUN apt update && apt install -y curl

COPY pyproject.toml poetry.lock ./

ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL $POETRY_URL | python3 -
RUN poetry config virtualenvs.create false

ENV APP /app
ENV PYTHONPATH $APP

RUN poetry install --no-cache --only main && \
    rm -rf /root/.cache/pypoetry && \
    rm -rf /root/.cache/pip

COPY . .

CMD ["python3", "."]
