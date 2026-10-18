# ClassGraph, containerised for the video-processing path.
#
# Scope, stated up front so nobody deploys this expecting the wrong thing:
# this image processes *recorded footage* and serves the dashboard. It does
# not do live webcam capture, because passing a camera into a container is
# platform-specific (a --device flag on Linux, not supported on Docker Desktop
# for Windows or macOS at all). For live sessions, install natively:
#     pip install -e ".[learned]" && classgraph serve
#
# Build:
#   docker build -t classgraph:1.0 .
#
# Process a folder of clips (read-only mount for the footage):
#   docker run --rm \
#     -v "$PWD/dataset:/data:ro" -v "$PWD/outputs:/app/outputs" \
#     classgraph:1.0 classgraph video --dir /data/clips --out /app/outputs/run
#
# Serve the dashboard:
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/outputs:/app/outputs" classgraph:1.0
#
# With a GPU (NVIDIA Container Toolkit required), add:  --gpus all

FROM python:3.10-slim

# OpenCV and MediaPipe link against system libraries that the slim image
# omits. Installing them here rather than falling back to opencv-headless
# keeps one code path for container and native installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, in their own layer, so editing source does not
# reinstall 2 GB of wheels on every build.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY tools/ ./tools/
COPY classgraph/ ./classgraph/
COPY labeling/ ./labeling/
COPY README.md ./

# The trained scorers are kilobytes and ship with the repo. Without them the
# pipeline silently falls back to the rule-based verdict, which measured 0.447
# against 0.650 for the model -- a quiet downgrade, so they are copied
# explicitly rather than left to a wildcard that might miss them.
COPY outputs/window_engagement_model.json outputs/mamba_attention_model.pt ./outputs/

RUN pip install --no-cache-dir -e .

# yolo11m.pt (~40 MB) is fetched by ultralytics on first use. Baking it in
# keeps the container able to start without network access, which is the
# normal expectation for a deployed image.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" || \
    echo "yolo11m.pt not pre-fetched; it will download on first run"

# Writable, and the mount point for results that should outlive the container.
RUN mkdir -p /app/outputs
VOLUME ["/app/outputs"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLASSGRAPH_HOST=0.0.0.0 \
    CLASSGRAPH_PORT=8000

EXPOSE 8000

# The dashboard answers /api/status without a session running, so this reports
# the process as unhealthy only when the server is genuinely not serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/status || exit 1

CMD ["classgraph", "serve", "--host", "0.0.0.0", "--port", "8000"]
