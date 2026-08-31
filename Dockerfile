FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Match pyproject.toml's bittensor range (the only dep needing a Rust toolchain)
RUN pip install --no-cache-dir "bittensor>=10.0.0,<11.0.0"

FROM python:3.10-slim

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN apt-get update && apt-get install -y --no-install-recommends libssl3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY fugal_subnet/ fugal_subnet/
COPY neurons/ neurons/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/

# CPU-only torch first (avoids pulling CUDA wheels), then the package itself,
# which installs the remaining pyproject dependencies.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

ENTRYPOINT ["python"]
