# Pin the base image (update deliberately; record the digest in your change log).
FROM dailyco/pipecat-base:latest

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HIPAA_MODE=1 \
    DATA_DIR=/data

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Application code (no secrets, no data — those come from env / the /data volume).
COPY --chown=1000:1000 . /app
WORKDIR /app

# Run as an unprivileged user; /data should be an encrypted volume mounted at runtime.
RUN mkdir -p /data && chown 1000:1000 /data && chmod 700 /data
USER 1000:1000
VOLUME ["/data"]
