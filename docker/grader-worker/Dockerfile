FROM python:3.10-slim@sha256:4101e4a31b9d2c927c9052ae2fa58b76a89265997ec9209c02933d9974214b80

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY --chmod=0555 runner.py /opt/fugal/runner.py
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "/opt/fugal/runner.py"]
