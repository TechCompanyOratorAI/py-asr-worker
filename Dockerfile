# =============================================================================
# ASR Worker — GPU (CUDA 12.1 + cuDNN 8) on Ubuntu 22.04
# =============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── Env ──────────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3-pip \
        python3.12-venv \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3       1 \
    && rm -rf /var/lib/apt/lists/*

# Verify FFmpeg
RUN ffmpeg -version && ffprobe -version

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 asrworker \
    && mkdir -p /app /tmp/asr-worker /app/logs /app/models \
    && chown -R asrworker:asrworker /app /tmp/asr-worker

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# Step 1: Install PyTorch with CUDA 12.1 GPU support FIRST (from official index)
RUN pip install --upgrade pip && \
    pip install \
        torch==2.5.1+cu121 \
        torchaudio==2.5.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install the rest of the dependencies (torch already satisfied above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY --chown=asrworker:asrworker . .

# Ensure ownership of dirs created after COPY
RUN chown -R asrworker:asrworker /app/logs /app/models

# ── Runtime ───────────────────────────────────────────────────────────────────
USER asrworker

# Health check — verify GPU is accessible
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; import sys; sys.exit(0)"

CMD ["python", "-u", "src/main.py"]
