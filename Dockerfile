# Production target is intentionally Linux amd64; keep this manifest digest and
# uv.lock review-coupled. The image contains no mutable dependency installation.
FROM ghcr.io/astral-sh/uv:0.8.14-python3.12-bookworm-slim@sha256:b748a09ec61c993083278020a84e16f0c81485c5349b30c64dd29d6a22da7462

# OCI labels. Without them the published package inherits the base image's
# description (uv's) and is not linked to this repository on ghcr.io. The
# labels are part of the image config and therefore of the digest; changing
# them is a new image, hence a new approved entry, like any other byte here.
LABEL org.opencontainers.image.source="https://github.com/fugal-ai/fugal-subnet" \
      org.opencontainers.image.title="fugal-miner" \
      org.opencontainers.image.description="Fugal subnet miner: trains cost-aware LLM router heads and serves attested benchmark proofs from a TDX confidential VM under dstack. Deploy by digest; see deploy/dstack/README.md." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="Fugal"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cache third-party wheels independently from project sources.
COPY pyproject.toml uv.lock README.md LICENSE ./
# --extra tee installs dcap-qvl, required for --live attestation verification.
RUN uv sync --frozen --no-dev --extra tee --no-install-project

COPY fugal_subnet/ fugal_subnet/
COPY neurons/ neurons/
# scripts/ is deliberately NOT copied: nothing inside the TD runs a script
# (the miner imports none; the operator-side tools run on the operator's
# machine), and every byte here is measured surface.
COPY data/ data/
RUN uv sync --frozen --no-dev --extra tee \
    && groupadd --gid 10001 fugal \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/fugal fugal \
    && mkdir -p /app/results /app/data /home/fugal/.bittensor \
    && chown -R 10001:10001 /app/results /app/data /home/fugal

USER 10001:10001

ENTRYPOINT ["/app/.venv/bin/python"]
