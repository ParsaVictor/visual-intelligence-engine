# CUDA runtime base: the models need a GPU to be practical.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/ParsaVictor/visual-intelligence-engine"
LABEL org.opencontainers.image.description="Multimodal search over an image archive"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libGL and libglib are required by opencv-python and absent from slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata first so the layer caches across source edits.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[models]"

COPY configs/ ./configs/
COPY scripts/ ./scripts/

# The gallery and index are mounted, never baked in: they hold biometric data.
VOLUME ["/app/data"]

ENTRYPOINT ["vie"]
CMD ["--help"]
