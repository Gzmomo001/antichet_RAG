# =============================================================================
# Makefile — Anti-Fraud RAG System
# =============================================================================
# Usage:  make <target>
# Run     `make help` to list all available commands.
#
# Requirements: Python 3.10+, uv, Docker & Docker Compose
# =============================================================================

.DEFAULT_GOAL := help

.PHONY: help install lint fmt test test-cov test-config test-schemas test-services clean ci

# -----------------------------------------------------------------------------
# Target: install
# Description: 安装项目依赖（使用 uv，速度更快）。
# When to use: 首次克隆或依赖变更后执行。
# Example: make install
# -----------------------------------------------------------------------------
install: ## Create venv and install dependencies
	uv sync --all-extras

# -----------------------------------------------------------------------------
# Target: lint
# Description: 检查代码规范（ruff check）。
# When to use: 提交前或 CI 流程中使用。
# Example: make lint
# -----------------------------------------------------------------------------
lint: ## Lint code with ruff
	uv run ruff check antifraud_rag tests

# -----------------------------------------------------------------------------
# Target: fmt
# Description: 自动格式化代码（ruff format + fix）。
# When to use: 开发过程中格式化代码。
# Example: make fmt
# -----------------------------------------------------------------------------
fmt: ## Format and fix code with ruff
	uv run ruff format antifraud_rag tests
	uv run ruff check --fix antifraud_rag tests

# -----------------------------------------------------------------------------
# Target: test
# Description: 运行测试套件（pytest）。
# When to use: 本地测试或 CI 流程中使用。
# Example: make test
#          make test ARGS="-v -k test_analyze"
#          make test-cov      # 运行测试并生成覆盖率报告
# -----------------------------------------------------------------------------
test: ## Run tests with pytest
	uv run pytest tests/ -v $(ARGS)

test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=antifraud_rag --cov-report=term-missing --cov-report=html $(ARGS)

test-config: ## Run config/settings tests only
	uv run pytest tests/test_config.py -v

test-schemas: ## Run schema validation tests only
	uv run pytest tests/test_schemas.py -v

test-services: ## Run service layer tests only
	uv run pytest tests/test_retrieval.py tests/test_embedding.py -v

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
