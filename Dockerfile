# InfiniteTalk - RunPod Serverless
# Updated: 2026-03-05 - fix: build < 30 min (remover flash-attn lento + torch reinstall)
# flash-attn é OPCIONAL: wan/modules/attention.py tem fallback para torch SDPA
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

# Base image já tem torch==2.6.0+cu124 — NÃO reinstalar (economiza ~4 min)

# xformers: wheel pré-compilada do PyTorch index (~43 MB, rápido)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-deps xformers==0.0.29.post3 \
      --index-url https://download.pytorch.org/whl/cu124

# flash-attn: PULADO — download de GitHub leva ~50 min (timeout de build é 30 min)
# O código já tem fallback automático para torch.nn.functional.scaled_dot_product_attention
# (ver wan/modules/attention.py linhas 157-188)

# Dependências Python
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
    pip install --no-cache-dir --no-deps yunchang distvae && \
    pip install --no-cache-dir "numpy>=1.23.5,<2"

# Baixar pesos no runtime (evita timeout no build)
# Models serão baixados no handler.py na primeira execução

CMD ["python", "-u", "handler.py"]
