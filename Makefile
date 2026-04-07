# =============================================================================
# Makefile — Anti-Fraud RAG System
# =============================================================================
# Usage:  make <target>
# Run     `make help` to list all available commands.
#
# Requirements: Python 3.10+, uv, Docker & Docker Compose
# =============================================================================

.DEFAULT_GOAL := help

.PHONY: help install dev docker-build docker-up docker-down docker-logs db-init lint fmt test test-cov test-unit test-config test-schemas test-services test-api clean ci

# -----------------------------------------------------------------------------
# Target: install
# Description: 安装项目依赖（使用 uv，速度更快）。
# When to use: 首次克隆或依赖变更后执行。
# Example: make install
# -----------------------------------------------------------------------------
install: ## Create venv and install dependencies
	uv venv
	uv pip install -e ".[dev]"

# -----------------------------------------------------------------------------
# Target: dev
# Description: 启动本地开发服务器（热重载）。
# When to use: 本地开发调试时使用。
# Note: 需先确保 .env 文件已配置或环境变量已导出。
# Example: make dev
# -----------------------------------------------------------------------------
dev: ## Start local dev server with hot reload
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# -----------------------------------------------------------------------------
# Target: docker-build
# Description: 构建 Docker 镜像（使用 uv 安装依赖）。
# When to use: 部署前或更新依赖后重新构建镜像。
# Example: make docker-build
# -----------------------------------------------------------------------------
docker-build: ## Build Docker image using uv
	docker compose build

# -----------------------------------------------------------------------------
# Target: docker-up
# Description: 启动所有 Docker Compose 服务（后台运行）。
# When to use: 本地容器化开发或部署启动。
# Note: 需确保 .env 文件已配置 API_KEY 等敏感变量。
# Example: make docker-up
# -----------------------------------------------------------------------------
docker-up: ## Start all services via docker compose
	docker compose up -d

# -----------------------------------------------------------------------------
# Target: docker-down
# Description: 停止并清理 docker compose 服务。
# When to use: 停止开发/测试环境时使用。
# Example: make docker-down
# -----------------------------------------------------------------------------
docker-down: ## Stop and remove docker compose services
	docker compose down

# -----------------------------------------------------------------------------
# Target: docker-logs
# Description: 实时查看 app 服务日志。
# When to use: 调试容器内服务行为时使用。
# Example: make docker-logs
#          make docker-logs SVC=db  # 查看 db 日志
# -----------------------------------------------------------------------------
docker-logs: ## Tail app logs (or SVC=db for database logs)
	docker compose logs -f $(SVC)

# -----------------------------------------------------------------------------
# Target: db-init
# Description: 初始化数据库（创建 extension + 表结构）。
# When to use: 首次启动数据库后，或数据库重置后执行。
# Note: DATABASE_URL 必须正确配置（Docker 环境会自动使用 compose 中的 db）。
# Example: make db-init
# -----------------------------------------------------------------------------
db-init: ## Initialize database (create pgvector extension + tables)
	uv run python scripts/init_db.py

# -----------------------------------------------------------------------------
# Target: lint
# Description: 检查代码规范（ruff check）。
# When to use: 提交前或 CI 流程中使用。
# Example: make lint
# -----------------------------------------------------------------------------
lint: ## Lint code with ruff
	ruff check .

# -----------------------------------------------------------------------------
# Target: fmt
# Description: 自动格式化代码（ruff format + fix）。
# When to use: 开发过程中格式化代码。
# Example: make fmt
# -----------------------------------------------------------------------------
fmt: ## Format and fix code with ruff
	ruff format .
	ruff check --fix .

# -----------------------------------------------------------------------------
# Target: test
# Description: 运行测试套件（pytest）。
# When to use: 本地测试或 CI 流程中使用。
# Example: make test
#          make test ARGS="-v -k test_analyze"
#          make test-cov      # 运行测试并生成覆盖率报告
#          make test-unit     # 仅运行单元测试
# -----------------------------------------------------------------------------
test: ## Run tests with pytest
	EMBEDDING_MODEL_URL="https://test.com" EMBEDDING_MODEL_API_KEY="test-key" pytest tests/ -v $(ARGS)

test-cov: ## Run tests with coverage report
	EMBEDDING_MODEL_URL="https://test.com" EMBEDDING_MODEL_API_KEY="test-key" pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html $(ARGS)

test-unit: ## Run unit tests only (excluding integration)
	EMBEDDING_MODEL_URL="https://test.com" EMBEDDING_MODEL_API_KEY="test-key" pytest tests/ -v -m "not integration" $(ARGS)

test-config: ## Run config/settings tests only
	EMBEDDING_MODEL_URL="https://test.com" EMBEDDING_MODEL_API_KEY="test-key" pytest tests/test_config.py -v

test-schemas: ## Run schema validation tests only
	EMBEDDING_MODEL_URL="https://test.com" EMBEDDING_MODEL_API_KEY="test-key" pytest tests/test_schemas.py -v

test-services: ## Run service layer tests only
	EMBEDDING_MODEL_URL="https://test.com" EMBEDDING_MODEL_API_KEY="test-key" pytest tests/test_retrieval.py tests/test_embedding.py -v

test-api: ## Run API endpoint tests only
	EMBEDDING_MODEL_URL="https://test.com" EMBEDDING_MODEL_API_KEY="test-key" pytest tests/test_analyze.py tests/test_data.py tests/test_main.py -v

# -----------------------------------------------------------------------------
# Target: clean
# Description: 清理构建产物和缓存文件。
# When to use: 清理环境或排查构建问题时使用。
# Example: make clean
# -----------------------------------------------------------------------------
clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# -----------------------------------------------------------------------------
# Target: ci
# Description: 运行完整 CI 流程（lint + test）。
# When to use: 本地模拟 CI 行为，或提交前检查。
# Example: make ci
# -----------------------------------------------------------------------------
ci: lint test ## Run lint and test suite (CI pipeline)

# -----------------------------------------------------------------------------
# Target: help
# Description: 显示所有可用命令及说明。
# Example: make help
# -----------------------------------------------------------------------------
help: ## Show this help message
	@echo ""
	@echo "  \033[36mCommon commands:\033[0m"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
