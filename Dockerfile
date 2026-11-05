# CUDA runtime base: the models need a GPU to be practical.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/ParsaVictor/visual-intelligence-engine"
LABEL org.opencontainers.image.description="Multimodal search over an image and video archive"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy dependency metadata first so the layer caches across source edits.
COPY pyproject.toml README.md requirements.txt ./
COPY src/ ./src/

# The pins first, so the extras below resolve to the tested versions.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir ".[models,gpu]"

# The index holds biometric identifiers, so nothing here should run as root.
RUN useradd --create-home --uid 10001 vie \
 && mkdir -p /app/data \
 && chown -R vie:vie /app
USER vie

COPY scripts/ ./scripts/

# The gallery and index are mounted, never baked in: they hold biometric data.
VOLUME ["/app/data"]

ENTRYPOINT ["vie"]
CMD ["doctor"]
