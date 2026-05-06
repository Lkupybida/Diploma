# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# ── System build tools ─────────────────────────────────────────────────────────
# gcc/g++ are needed by some Python packages that compile C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~4 GB vs the CUDA build).
# Must come before chronos-forecasting / pytorch-forecasting so they don't
# pull in the heavier CUDA wheel as a transitive dep.
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# Copy everything in the interface/ directory (Python scripts + models/).
COPY . .

# weather.py's smart_requests() reads this file on every iteration after the
# first, to check whether the user has requested an early stop.  An empty file
# means "continue normally".  Pre-create it so the runtime never hits a
# FileNotFoundError.
RUN printf '' > weather_func_status.txt

# Create the mount-point tree expected by the inference scripts
# ("../data/..." relative to /app → /data/...).
# These directories are empty inside the image; the real files arrive via
# the bind-mounts defined in docker-compose.yml.
RUN mkdir -p \
        /data/additional_data \
        /data/raw \
        /data/inference/bronze \
        /data/inference/silver \
        /data/inference/gold

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8501

# Lightweight health-check using Python's stdlib (no curl required in the image).
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]