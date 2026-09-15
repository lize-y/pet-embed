"""pet-embed：单图片接口，输出画面里每只宠物的位置 + 384 维归一化特征向量。

职责单一：检测 + 向量化。跟踪、检索、标注由调用方负责。
"""
import io
import logging
import os

# 必须先于 transformers / ultralytics import（HF_ENDPOINT 注入要在它们读环境变量之前）
from app.config import settings

import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO

# 特征提取：DINOv2-Small 宠物个体识别微调版，输出 384 维 CLS 特征。
# 默认加载本地 models/dinov2（构建期烘焙，完全离线），也可用 FEATURE_MODEL 指定 HF 仓库名
FEATURE_MODEL = os.environ.get("FEATURE_MODEL", "models/dinov2")
feature_processor = AutoImageProcessor.from_pretrained(FEATURE_MODEL)
feature_model = AutoModel.from_pretrained(FEATURE_MODEL)
feature_model.eval()

# 检测：YOLOv8n，轻量，CPU 友好。优先用本地权重，避免运行时从 GitHub 下载失败
# （受限网络下 HF 走镜像可用，GitHub 不通，构建期把 yolov8n.pt 烘焙进镜像）
detector = YOLO(os.environ.get("YOLO_WEIGHTS", "models/yolov8n.pt"))

# COCO: cat=15, dog=16
PET_CLASS_IDS = {15: "cat", 16: "dog"}

torch.set_num_threads(settings.torch_threads)

app = FastAPI(title="pet-embed", version="0.1.0")

if not settings.auth_enabled:
    logging.getLogger("uvicorn.error").warning(
        "AUTH_ENABLED=false：接口无鉴权保护，请勿暴露到不受信网络！"
    )


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """鉴权：X-API-Key 必须等于 .env 里配置的固定 key。"""
    if not settings.auth_enabled:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="无效的 API Key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _detect(image: Image.Image):
    """YOLO 检测，返回单个 Results。同步函数，供线程池调用。"""
    return detector(image, conf=settings.min_conf, verbose=False)[0]


@app.post("/detect")
async def detect_pets(
    _: None = Depends(require_api_key),
    file: UploadFile = File(...),
) -> dict:
    try:
        contents = await file.read()
        if len(contents) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="图片过大")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="无法处理的图片文件")

    # YOLO 检测：同步推理丢线程池，不阻塞事件循环
    results = await run_in_threadpool(_detect, image)
    boxes = results.boxes

    # 无检测时 results.boxes 可能为 None（跨版本行为不一），统一当空处理
    if boxes is None:
        return {"result": [], "count": 0}

    # 过滤出宠物框 + 收集裁剪图（一次性准备，便于批量 forward）
    w, h = image.size
    crops, metas = [], []
    for box in boxes:
        cls_id = int(box.cls.item())
        if cls_id not in PET_CLASS_IDS:
            continue

        conf = float(box.conf.item())
        if conf < settings.min_conf:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_w, box_h = x2 - x1, y2 - y1

        # 裁剪：留边距；square_crop 时按框中心取方形，避免 DINOv2 resize 拉伸
        if settings.square_crop:
            side = max(box_w, box_h) * (1 + 2 * settings.crop_pad)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx1, cy1 = max(0.0, cx - side / 2), max(0.0, cy - side / 2)
            cx2, cy2 = min(float(w), cx + side / 2), min(float(h), cy + side / 2)
        else:
            cx1 = max(0.0, x1 - box_w * settings.crop_pad)
            cy1 = max(0.0, y1 - box_h * settings.crop_pad)
            cx2 = min(float(w), x2 + box_w * settings.crop_pad)
            cy2 = min(float(h), y2 + box_h * settings.crop_pad)

        # 钳制最小裁剪尺寸，防止贴边/极小框 clamp 后出现空图或负尺寸
        if cx2 - cx1 < 1 or cy2 - cy1 < 1:
            continue

        crops.append(image.crop((cx1, cy1, cx2, cy2)))
        metas.append({
            "box": {
                "x_min": round(x1, 2),
                "y_min": round(y1, 2),
                "x_max": round(x2, 2),
                "y_max": round(y2, 2),
                "probability": round(conf, 4),
            },
            "label": PET_CLASS_IDS[cls_id],
        })

    if not crops:
        return {"result": [], "count": 0}

    # DINOv2 批量提特征 + L2 归一化（N 只宠物一次 forward）
    feat_inputs = feature_processor(images=crops, return_tensors="pt")
    with torch.no_grad():
        cls_feat = feature_model(**feat_inputs).last_hidden_state[:, 0]
        cls_feat = F.normalize(cls_feat, p=2, dim=-1)
        embeddings = cls_feat.tolist()

    detections = [
        {**meta, "embedding": emb}
        for meta, emb in zip(metas, embeddings)
    ]

    return {"result": detections, "count": len(detections)}
