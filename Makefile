# pet-embed 任务编排：依赖 / 模型 / 本地运行 / 镜像构建 / 测试

.PHONY: sync models setup genenv run test test-gen build up start down logs clean

# ---------- 依赖与模型 ----------

## 安装 Python 依赖（CPU torch，从 uv.lock 锁定版本）
sync:
	uv sync

## 下载模型权重到 models/（走 hf-mirror，幂等可重复；本地运行时用）
models:
	HF_ENDPOINT=$${HF_ENDPOINT:-https://hf-mirror.com} uv run python scripts/download_models.py

## 从 .env.example 生成 .env，API_KEY 随机生成（已存在则跳过）
genenv:
	@if [ -f .env ]; then \
		echo ".env 已存在，跳过生成（如需重置先删除）"; \
	else \
		key=$$(openssl rand -hex 24); \
		sed "s/^API_KEY=.*/API_KEY=$$key/" .env.example > .env; \
		echo "已生成 .env，API_KEY=$$key"; \
	fi

## 一条命令完成配置 + 依赖 + 模型（拉取项目后第一步）
setup: genenv sync models

# ---------- 本地运行 ----------

## 启动服务（前台，Ctrl+C 停止）
run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8001}

## 健康检查
health:
	curl -s http://127.0.0.1:$${PORT:-8001}/health

## 检测一张图：make detect IMG=test.jpg KEY=yourkey
detect:
	curl -s -H "X-API-Key: $${KEY:-change-me}" -X POST -F "file=@$${IMG}" http://127.0.0.1:$${PORT:-8001}/detect

# ---------- Docker 镜像 ----------

## 构建镜像（uv 从 lock 装依赖 + 模型自动本地优先/下载，clone 即用）
build:
	docker compose build

## 启动容器（仅启动，不构建；API_KEY 未设置或仍是 change-me 时拒绝启动）
up:
	@key=$${API_KEY:-$$(sed -n 's/^API_KEY=//p' .env 2>/dev/null)}; \
	if [ -z "$$key" ] || [ "$$key" = "change-me" ]; then \
		echo "ERROR: API_KEY 未设置或仍是默认值 change-me。请先: make genenv 生成 .env 并设置强随机值"; exit 1; \
	fi
	docker compose up -d

## 一键启动：生成 .env（如无）→ 构建镜像 → 启动容器
start: genenv build up

## 停止容器
down:
	docker compose down

## 查看容器日志
logs:
	docker compose logs -f

# ---------- 验证 ----------

## 跑一遍接口自测（需服务在运行）
test:
	@test -f test_images/real_cat.jpg || (echo "缺少 test_images/real_cat.jpg（git 忽略，需自备或 make test-gen）"; exit 1)
	@echo "== health ==" && curl -s http://127.0.0.1:$${PORT:-8001}/health && echo
	@echo "== detect (cat) ==" && curl -s -H "X-API-Key: $${KEY:-change-me}" -X POST -F "file=@test_images/real_cat.jpg" http://127.0.0.1:$${PORT:-8001}/detect | head -c 200 && echo

## 生成测试图（合成猫图，供 make test 用）
test-gen:
	@mkdir -p test_images
	@uv run python -c "from PIL import Image, ImageDraw; img=Image.new('RGB',(640,480),(200,210,220)); d=ImageDraw.Draw(img); d.ellipse([180,150,460,430],fill=(120,90,70)); d.ellipse([260,60,380,200],fill=(120,90,70)); d.polygon([(300,60),(285,15),(330,45)],fill=(120,90,70)); d.polygon([(340,60),(355,15),(370,45)],fill=(120,90,70)); d.ellipse([285,120,305,140],fill=(30,30,30)); d.ellipse([340,120,360,140],fill=(30,30,30)); d.ellipse([330,160,360,185],fill=(255,150,150)); img.save('test_images/real_cat.jpg'); print('generated test_images/real_cat.jpg')"

## 清理运行时产物（不动 models/，除非 FORCE=1）
clean:
	rm -rf runs/ datasets/ weights/
	@if [ "$${FORCE}" = "1" ]; then rm -rf .venv/ models/; echo "removed .venv/ and models/ (FORCE=1)"; fi

help:
	@grep -E "^[a-zA-Z-]+:" Makefile | sed 's/:.*//' | sort -u
