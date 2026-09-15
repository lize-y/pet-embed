# syntax=docker/dockerfile:1
# 单构建路径：uv 从 uv.lock 安装依赖（clone 即用），模型本地优先/缺失下载。
# 多阶段：deps 阶段只产出 site-packages，final 从 base 起步，uv 二进制/lock/缓存不进最终镜像。

# ---------- 基础阶段（公共层） ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    FEATURE_MODEL=models/dinov2 \
    YOLO_WEIGHTS=models/yolov8n.pt \
    HF_HUB_OFFLINE=1

RUN useradd -u 10001 appuser

# opencv-python 需要 libgl1/libglib2.0-0，torch 需要 libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 依赖：uv 从 lock 安装到系统 site-packages ----------
# 用 export + pip install --system 直接装系统目录，不建 .venv：
# 避免 .venv 的 _virtualenv.pth / distutils-precedence.pth 被复制进系统
# site-packages 后在 Python 启动早期破坏 importlib 初始化（启动崩溃）。
FROM base AS deps
COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/
COPY --chown=appuser:appuser pyproject.toml uv.lock /app/
RUN uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-cache \
        --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# ---------- 最终镜像：从 base 起步，只 COPY 依赖 ----------
FROM base AS final

COPY --from=deps /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# 防御性清理：确保没有任何 .pth 残留（uv pip install --system 本身不产生，
# 但保留作为保险）。.pth 的唯一作用是启动时加路径/执行代码，运行时不需要。
RUN find /usr/local/lib/python3.11/site-packages -maxdepth 1 \
        \( -name '*.pth' -o -name '_virtualenv.py' \) -delete

# 精简运行时依赖：polars(+runtime) 是 ultralytics 的可选传递依赖（~172M），
# 仅在 plotting/benchmarks 等函数内局部 import，推理路径不触发，删掉减小镜像
RUN rm -rf /usr/local/lib/python3.11/site-packages/polars \
    /usr/local/lib/python3.11/site-packages/polars_runtime_32 \
    /usr/local/lib/python3.11/site-packages/polars-*.dist-info \
    /usr/local/lib/python3.11/site-packages/polars_runtime_32-*.dist-info

# ---------- 模型：本地优先，缺失回退下载 ----------
# 放在 app/ 之前：模型不常变，避免改代码触发重新拷贝/下载（层缓存）
# models/.gitkeep 始终存在（git 跟踪），所以 COPY models/ 永不失败
# 本地有实际权重（非空）→ 用本地；只有 .gitkeep（clone 场景）→ 容器内下载
# scripts/ 必须先于下载逻辑 COPY（download_models.py 在那里）
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser models/ /tmp/models-src/
RUN if [ "$(find /tmp/models-src -type f ! -name '.gitkeep' | wc -l)" -gt 0 ]; then \
        echo "using LOCAL models" && \
        mkdir -p /app/models && cp -r /tmp/models-src/. /app/models/; \
    else \
        echo "downloading models (hf-mirror)..." && \
        HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 python scripts/download_models.py; \
    fi
RUN rm -rf /tmp/models-src

# 构建期 HF 缓存已冗余（模型在 /app/models），清理。
# 下载发生在 USER appuser 之前（root，HOME=/root），缓存落在 /root/.cache；
# useradd 无 -m 也不存在 /home/appuser，一并清理无副作用。
RUN rm -rf /root/.cache/huggingface /home/appuser/.cache/huggingface 2>/dev/null || true

# ---------- 应用代码：常变，放最后（层缓存最大化） ----------
COPY --chown=appuser:appuser app/ /app/app/

USER appuser

EXPOSE 8001

# 用 python urllib 做健康检查，省掉 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
