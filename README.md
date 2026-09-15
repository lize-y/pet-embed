# pet-embed

单图片接口：检测画面中的宠物（cat/dog），输出每只宠物的位置框 + **384 维 L2 归一化特征向量**。

职责单一：**检测 + 向量化**。跟踪、检索、标注由调用方负责。

## 架构

```
                 POST /detect (multipart: file + X-API-Key)
                              │
                              ▼
        ┌───────────────────────────────┐
        │  1. YOLOv8n 目标检测           │  → 定位 cat/dog 框 (x1,y1,x2,y2,conf)
        │     COCO 80 类，仅放行 cat/dog │     其他 74 类跳过
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │  2. 裁剪 + 方形化             │  → 按框中心取方形裁剪（默认），
        │     5% 边距                   │     避免 DINOv2 resize 拉伸变形
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │  3. DINOv2-Small 提特征       │  → 批量 forward，取 CLS token (384 维)
        │     宠物个体识别微调版         │     L2 归一化 → 模长 = 1
        └──────────────┬────────────────┘
                       ▼
              { result: [...], count: N }
```

## 技术栈

| 组件 | 用途 | 模型 |
|---|---|---|
| YOLOv8n | 目标检测（cat/dog） | `models/yolov8n.pt` (COCO, 6.5MB) |
| DINOv2-Small | 特征提取（个体识别） | `models/dinov2/` (AvitoTech 微调版, 88MB) |
| FastAPI + uvicorn | HTTP 服务 | — |

**模型说明**：DINOv2 是 `AvitoTech/DINO-v2-small-for-animal-identification`，专门为**猫狗个体识别**微调（re-ID）。对 cat/dog 的区分能力有保障；其他动物（bird/horse 等 COCO 有但被过滤）个体区分无保证。

## 接口契约

### POST `/detect`

`multipart/form-data`，字段 `file`。请求头需带 `X-API-Key`。

**请求**

```bash
curl -X POST \
  -H "X-API-Key: <key>" \
  -F "file=@test.jpg" \
  http://<host>:8001/detect
```

**响应 200**

```json
{
  "result": [
    {
      "box": {
        "x_min": 104.04,
        "y_min": 35.43,
        "x_max": 465.34,
        "y_max": 423.78,
        "probability": 0.9111
      },
      "label": "cat",
      "embedding": [384 个浮点数，L2 归一化，模长 ≈ 1]
    }
  ],
  "count": 1
}
```

| 字段 | 说明 |
|---|---|
| `box.x_min/y_min/x_max/y_max` | 像素坐标，对应**原图尺寸** |
| `box.probability` | YOLO 检测置信度（≥ `MIN_CONF`） |
| `label` | `cat` 或 `dog` |
| `embedding` | 384 维，模长 ≈ 1，**余弦相似度 = 点积**（无需再归一化） |

**错误码**

| 状态 | 场景 | 响应体 |
|---|---|---|
| 200 | 正常（含无宠物） | `{"result":[], "count":0}` |
| 400 | 图片损坏/非图片 | `{"detail":"无法处理的图片文件"}` |
| 401 | 无/错 `X-API-Key` | `{"detail":"无效的 API Key"}` |
| 413 | 图片超过 `MAX_IMAGE_BYTES` | `{"detail":"图片过大"}` |

### GET `/health`

```json
{ "status": "ok" }
```

## 快速开始（本地）

前置：已安装 [uv](https://docs.astral.sh/uv/)、[make](https://www.gnu.org/software/make/)、Python 3.11。

**一条命令装依赖 + 下模型**（拉取项目后第一步）：

```bash
make setup        # = make genenv + make sync + make models
```

等价手动执行：

```bash
make genenv                 # 1. 生成 .env（API_KEY 随机生成）
uv sync                     # 2. 装依赖（CPU torch，约 1.5GB，走官方 pytorch CPU 源）
make models                 # 3. 下载模型（91MB，走 hf-mirror，幂等可重复）
make run                    # 4. 启动（uvicorn 0.0.0.0:8001）
```

测试：

```bash
make health
make detect IMG=test.jpg KEY=你的key   # KEY 需与 .env 里的 API_KEY 一致（make genenv 自动生成）
```

## Makefile 命令一览

```bash
make setup      # 本地开发：配置 + 依赖 + 模型（genenv + sync + models）
make genenv     # 从 .env.example 生成 .env（API_KEY 随机生成，已存在则跳过）
make sync       # 仅装依赖
make models     # 仅下载模型（幂等）
make run        # 本地启动服务（PORT 可改）
make health     # 健康检查
make detect IMG=x.jpg KEY=y   # 检测一张图
make build      # 仅构建镜像（uv 装依赖 + 下载模型，无需 env）
make start      # 一键启动：genenv → build → up（clone 即用，推荐）
make up         # 仅启动容器（不构建；API_KEY 校验）
make down       # 停止容器
make logs       # 容器日志
make test       # 接口自测（需服务运行中）
make clean      # 清运行时产物（FORCE=1 连 .venv/models 一起删）
make help       # 列出所有目标
```

## Docker 部署

### 一键启动（clone 即用，推荐）

```bash
make start      # = genenv + build + up
```

`make start` 自动完成三件事：
1. **`make genenv`** — 生成 `.env`（API_KEY 随机；已存在则跳过，不覆盖）
2. **`make build`** — 构建镜像（容器内装依赖 + 模型本地优先/下载，无需 env）
3. **`make up`** — 校验 API_KEY 后启动容器

> **API_KEY 必填（启动时校验）**：构建镜像不需要 API_KEY（环境变量不进镜像，`make build` 可独立跑）；`make up` 启动容器前会校验——未设置或仍是默认 `change-me` 直接拒绝启动（防止对外鉴权用公开默认值）。`make start` 里的 genenv 会自动生成随机 key。

### 手动分步（构建 / 启动分离）

```bash
make build      # 1. 仅构建镜像（无需 env）
make genenv     # 2. 生成 .env（API_KEY 随机）
make up         # 3. 仅启动容器（校验 API_KEY）
```

### 构建策略：依赖重装，模型可拷

| 资源 | 处理方式 | 原因 |
|---|---|---|
| **依赖** | 构建期用 uv 从 `uv.lock` **重新安装** | 依赖是可执行代码，绑定架构/ABI/路径，本地 `.venv` 不能跨环境直接用（含 `_virtualenv` 钩子，COPY 进系统 site-packages 会破坏 Python 启动） |
| **模型** | 本地 `models/` 有 → COPY；没有 → 构建期下载 | 模型是纯数据（权重矩阵），跨架构通用，可安全 COPY |

```bash
# 构建镜像（依赖 uv 安装，模型本地优先/缺失下载）
make build
```

> **模型自动检测**：`models/` 下 `.gitkeep` 始终存在（git 跟踪），所以 COPY 永不失败；只有 `.gitkeep`（clone）时自动触发下载。

### 多阶段结构（uv 不进最终镜像）

```
base ── deps       (uv sync → site-packages；uv 二进制/缓存留在此层)
   └── final       (from base 起步，只 COPY site-packages)
        ├── 删 _virtualenv 钩子（防 Python 启动崩溃）
        ├── 删 polars（ultralytics 可选依赖，省 ~172M）
        ├── COPY app/ scripts/
        ├── 模型：本地优先 / 下载兜底
        └── 清 HF 缓存（/root 路径，避免模型存两份）
```

最终镜像**不含** uv 二进制、lock 文件、安装缓存——只有 Python 依赖 + 代码 + 模型。

### 镜像特性

- **模型已烘焙**（`models/` 91MB），运行时**完全离线**，无需联网
- 非 root 用户（`appuser`, UID 10001）
- 系统 site-packages（无 `PYTHONPATH` 依赖，`.pth` 正常）
- healthcheck 用 Python urllib（不装 curl）
- 无 uv / pip / lock 文件残留

## 配置（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_KEY` | `change-me`（compose 默认，`make up` 拒绝启动） | 调用方 `X-API-Key` 必须匹配。部署时设置强随机值 |
| `AUTH_ENABLED` | `true` | 置 `false` 临时关掉鉴权（调试用） |
| `TORCH_THREADS` | `4` | CPU 推理线程数 |
| `MIN_CONF` | `0.25` | YOLO 检测置信度下限（直接传给 YOLO，低于此值不返回） |
| `CROP_PAD` | `0.05` | 裁剪边距比例（相对框尺寸） |
| `SQUARE_CROP` | `true` | 按框中心取方形裁剪，避免 DINOv2 resize 拉伸变形 |
| `MAX_IMAGE_BYTES` | `20971520` | 请求图片大小上限（字节，20MB），超限返回 413 |

> `HF_ENDPOINT`、`FEATURE_MODEL`、`YOLO_WEIGHTS` 为**构建期/本地专用**变量，已烘焙进镜像（Dockerfile `ENV`），运行时**无需**在 `.env` 设置。仅当本地 `uv run` 调试需覆盖路径/镜像时使用（见 `.env.example` 底部注释）。

## 调用方对接

### Python

```python
import requests

resp = requests.post(
    "http://<host>:8001/detect",
    headers={"X-API-Key": "<key>"},
    files={"file": open("frame.jpg", "rb")},
)
data = resp.json()
for det in data["result"]:
    box = det["box"]
    emb = det["embedding"]   # numpy 数组，L2 归一化，直接用点积做相似度
    label = det["label"]
    # 1. 用 box 画框
    # 2. 用 emb 去向量库检索 → 得到宠物身份
    # 3. 把身份缓存到跟踪 ID 上
```

### 向量检索注意

- `embedding` 已是单位向量，向量库用**余弦相似度 / 内积**均可
- 与库中已存向量对比时，直接 `np.dot(a, b)` 即得余弦相似度
- 相似度阈值建议先跑真实数据标定（猫狗 re-ID 通常 0.7–0.85 之间分界）

## 性能（CPU）

- **单图单宠物**：约 200–400ms（YOLO ~50–150ms + DINOv2 ~80–200ms）
- **多宠物**：DINOv2 **批量 forward**（N 只一次推理），比逐个快
- **内存**：约 1.5GB（torch + 两个模型）
- **并发**：YOLO/DINOv2 是全局单例，CPU 单线程推理，并发请求串行执行

## 边界情况

| 情况 | 行为 |
|---|---|
| 图中无宠物 | `result: []`, `count: 0` |
| 图片损坏 | 400 |
| 宠物太小（< ~30px） | YOLO 可能漏检；可调低 `MIN_CONF`，但会引入误检 |
| 多只宠物 | `result` 数组多项，按 YOLO 输出顺序 |
| 置信度过低检测 | 被 `MIN_CONF` 过滤 |

## 常见问题

**Q: 模型文件在哪？为什么不进 git？**
`models/`（91MB）是构建产物，已在 `.gitignore` 忽略。克隆仓库后 `models/` 只有 `.gitkeep` 占位。**镜像构建会自动下载模型**（`make build` / `make start` 内置），本地运行时才需 `make models` 手动下载。`.dockerignore` **不**忽略 models（本地有模型时构建期 COPY 进镜像）。

**Q: 服务启动慢/卡住？**
启动时加载两个模型约 3–5 秒。若卡在 `huggingface.co`，说明模型未烘焙且 `HF_ENDPOINT` 未生效（本地 `uv run` 调试时）——`make genenv` 后确认 `.env` 底部注释的 `HF_ENDPOINT` 已取消注释，且 `app.config` 先于 `transformers` import。

**Q: 换 GPU 怎么改？**
`pyproject.toml` 里 `[[tool.uv.index]]` 的 `cpu-torch` 换成 CUDA 索引，重新 `uv sync`；Dockerfile 基础镜像换 `nvidia/cuda`。推理代码无需改动（torch 自动检测设备）。

**Q: 想支持更多宠物类别？**
改 `app/main.py` 的 `PET_CLASS_IDS`（COCO 里 bird=14、horse=17 等）。但 DINOv2 个体识别只对猫狗有保障。

## 项目结构

```
pet-embed/
├── app/
│   ├── __init__.py
│   ├── main.py          # /detect + /health 接口
│   └── config.py        # .env 配置（pydantic-settings）
├── scripts/
│   └── download_models.py  # 模型下载（走 hf-mirror，幂等）
├── models/              # 模型权重（git 忽略，仅 .gitkeep 占位）
│   ├── dinov2/          # DINOv2 特征模型
│   └── yolov8n.pt       # YOLO 检测权重
├── Makefile             # 任务编排：genenv/setup/build/start/up 等
├── Dockerfile           # 多阶段：deps(uv装依赖) + final(运行时)
├── docker-compose.yml
├── pyproject.toml       # uv 配置 + CPU torch 源
├── uv.lock              # 锁文件（应提交）
├── .env.example
├── .gitignore
└── .dockerignore
```
