"""pet-embed 服务配置：从 .env 读取，可用环境变量覆盖。"""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = "change-me"        # 调用方请求头 X-API-Key 必须匹配
    auth_enabled: bool = True         # 调试时可置 False 临时关掉鉴权
    torch_threads: int = 4            # CPU 推理线程数
    min_conf: float = 0.25            # YOLO 检测置信度下限（低于此值不返回）
    crop_pad: float = 0.05            # 裁剪边距比例（相对框尺寸）
    square_crop: bool = True          # 按框中心取方形裁剪，避免 DINOv2 拉伸变形
    max_image_bytes: int = 20 * 1024 * 1024  # 请求图片大小上限（20MB）
    hf_endpoint: str = ""             # 仅本地 uv run 调试需要联网时用；镜像运行时已烘焙 HF_ENDPOINT

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# transformers / huggingface_hub 靠 HF_ENDPOINT 环境变量找镜像。
# 必须在模型加载（app.main import）之前注入环境变量。
if settings.hf_endpoint:
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
