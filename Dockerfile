# Environment used to synthesise and score the paper's audio (NVIDIA A10G, driver CUDA 12.4).
# Build:  docker build -t condsites .
# Run:    docker run --gpus all --shm-size 8g -v "$PWD":/work -w /work condsites bash scripts/setup_all.sh
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq \
 && apt-get install -y -qq git git-lfs sox ffmpeg build-essential curl unzip wget libsndfile1 \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install --skip-repo
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
 && uv python install 3.10 3.11
