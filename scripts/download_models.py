"""下载模型权重到 models/（git 忽略，构建期烘焙进镜像）。

走 HF 镜像（HF_ENDPOINT），不依赖 GitHub。可重复执行（已下载的复用缓存）。
"""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

FEATURE_REPO = "AvitoTech/DINO-v2-small-for-animal-identification"
FEATURE_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
DETECTOR_REPO = "Ultralytics/YOLOv8"
DETECTOR_FILE = "yolov8n.pt"

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def _download(repo: str, filename: str, dest: Path) -> None:
    hf_hub_download(repo, filename, local_dir=dest)
    # hf_hub_download(local_dir=...) 会留 .cache，清掉保持目录干净
    cache = dest / ".cache"
    if cache.is_dir():
        shutil.rmtree(cache)
    print(f"ok {dest / filename}")


def main() -> None:
    MODELS.mkdir(exist_ok=True)

    feature_dir = MODELS / "dinov2"
    feature_dir.mkdir(exist_ok=True)
    for f in FEATURE_FILES:
        _download(FEATURE_REPO, f, feature_dir)

    _download(DETECTOR_REPO, DETECTOR_FILE, MODELS)


if __name__ == "__main__":
    main()
