# syntax=docker/dockerfile:1

# 3.12 rather than latest: pyproject pins <3.14, and the ONNX runtime that
# fastembed pulls in is slowest to publish wheels for a brand-new Python.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Only the dependencies are installed, not the project itself. WORKDIR is
# /app and the code lands at /app/app, so `import app` already resolves - and
# building the wheel here would need app/ copied first, which would throw the
# dependency layer away on every edit to the source.
COPY pyproject.toml ./
RUN python -c "import tomllib, pathlib; \
print('\n'.join(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['dependencies']))" \
      > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ ./app/
COPY widget/ ./widget/

# The database lives here. Mount a volume on it: an ephemeral filesystem loses
# every Property, origin allowlist and spend ledger on restart, which takes
# every client's widget down with it.
RUN mkdir -p /app/data

# Not root. Nothing needs to write outside /app/data.
RUN useradd --create-home --uid 10001 guide && chown -R guide:guide /app
USER guide

EXPOSE 8000

# $PORT is set by the platform; 8000 is the fallback for a plain `docker run`.
# One worker on purpose: the rate limiter and the usage limiter are both
# per-process, so a second worker silently doubles both ceilings.
# keep-alive is above the answer latency so a streamed turn is never cut off.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 65"]
