import runpod
import os
import time
import requests
import subprocess
import uuid
import base64
import json
from pathlib import Path

WORKSPACE = Path("/workspace")
OUTPUT_DIR = WORKSPACE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR = WORKSPACE / "weights"


def download_file(url: str, dest: Path) -> Path:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)
    return dest


def handler(job):
    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    audio_url = job_input.get("audio_url")
    prompt = job_input.get("prompt", "")

    if not image_url:
        return {"error": "image_url é obrigatório"}
    if not audio_url:
        return {"error": "audio_url é obrigatório"}

    job_id = job.get("id", str(uuid.uuid4()))
    tmp_dir = OUTPUT_DIR / job_id
    tmp_dir.mkdir(exist_ok=True)

    try:
        print(f"[InfiniteTalk] Baixando imagem: {image_url}")
        image_path = download_file(image_url, tmp_dir / "input.png")

        print(f"[InfiniteTalk] Baixando áudio: {audio_url}")
        audio_path = download_file(audio_url, tmp_dir / "input.wav")

        # Criar JSON de input
        input_json = {
            "image": str(image_path),
            "audio": str(audio_path),
            "prompt": prompt,
        }
        json_path = tmp_dir / "input.json"
        with open(json_path, "w") as f:
            json.dump(input_json, f)

        print(f"[InfiniteTalk] Iniciando inferência...")
        start_time = time.time()

        cmd = [
            "python3", "generate_infinitetalk.py",
            "--ckpt_dir", str(WEIGHTS_DIR / "Wan2.1-I2V-14B-480P"),
            "--wav2vec_dir", str(WEIGHTS_DIR / "chinese-wav2vec2-base"),
            "--infinitetalk_dir", str(WEIGHTS_DIR / "InfiniteTalk" / "single" / "infinitetalk.safetensors"),
            "--input_json", str(json_path),
            "--output_dir", str(tmp_dir),
            "--task", "infinitetalk-14B",
            "--size", "infinitetalk-480",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE))

        if proc.returncode != 0:
            raise RuntimeError(f"Inference failed:\n{proc.stderr}")

        elapsed = round(time.time() - start_time, 2)
        print(f"[InfiniteTalk] Concluído em {elapsed}s")

        output_files = list(tmp_dir.glob("*.mp4"))
        if not output_files:
            raise RuntimeError(f"Nenhum vídeo gerado em {tmp_dir}")

        with open(output_files[0], "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "model": "InfiniteTalk",
            "execution_time_seconds": elapsed,
            "video_base64": video_b64,
        }

    except Exception as e:
        return {"error": str(e), "model": "InfiniteTalk"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
