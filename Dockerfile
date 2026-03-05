# InfiniteTalk - RunPod Serverless
# Updated: 2026-03-05 - fix: ordem correta torch→xformers→flash-attn + wheel pré-compilada
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/workspace/output \
    PYTHONPATH=/workspace \
    WAN_GPU_COUNT=0 \
    HF_HUB_DISABLE_XET=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1-mesa-glx libglib2.0-0 wget curl git-lfs espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

# Corrigir compatibilidade Python 3.11
RUN sed -i "s/from inspect import ArgSpec/# from inspect import ArgSpec  # Python 3.11 fix/" wan/multitalk.py 2>/dev/null || true

# PASSO 1 — Garantir torch 2.6.0+cu124 ANTES de qualquer extensão CUDA
# (base image já vem com 2.6.0; reinstalamos para garantir caso deps abaixo alterem)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# PASSO 2 — Instalar xformers (wheel pré-compilada para torch 2.6.0+cu124)
RUN pip install --no-cache-dir --no-deps xformers==0.0.29.post3 \
      --index-url https://download.pytorch.org/whl/cu124

# PASSO 3 — Instalar flash-attn com wheel pré-compilada (evita ~30 min de compilação CUDA)
# Wheel: Python 3.11 + CUDA 12.4 + Torch 2.6, sem cxx11-abi
RUN pip install --no-cache-dir ninja psutil packaging wheel && \
    pip install --no-cache-dir \
      "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# PASSO 4 — Instalar demais dependências Python
RUN pip install --no-cache-dir \
    runpod==1.6.2 \
    huggingface_hub \
    requests \
    "opencv-python>=4.9.0.80" \
    "diffusers>=0.31.0" \
    "transformers>=4.49.0,<5.0.0" \
    "tokenizers>=0.20.3" \
    "accelerate>=1.1.1" \
    tqdm imageio easydict ftfy \
    imageio-ffmpeg scikit-image loguru \
    "numpy>=1.23.5,<2" \
    pyloudnorm "optimum-quanto==0.2.6" \
    "scenedetect" "moviepy==1.0.3" decord \
    einops sentencepiece librosa soundfile \
    "misaki[en]" && \
    pip install --no-cache-dir --no-deps "xfuser>=0.4.1" && \
    pip install --no-cache-dir --no-deps yunchang distvae

# PASSO 5 — Re-pinnar torch (caso alguma dep acima tenha alterado)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir "numpy>=1.23.5,<2"

# Baixar pesos no runtime (evita timeout no build)
# Models serão baixados no handler.py na primeira execução

CMD ["python", "-u", "handler.py"]
