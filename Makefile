# Developer entrypoints. `make help` lists everything.
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create .venv and install the exact dependency set pinned in uv.lock
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --locked --extra dev --extra ui; \
	else \
		echo ">> uv not found - falling back to pip."; \
		echo ">> Versions will be re-resolved and may NOT match uv.lock."; \
		echo ">> Install uv for reproducible builds: https://docs.astral.sh/uv/"; \
		$(PY) -m venv $(VENV); \
		$(BIN)/python -m pip install --upgrade pip; \
		$(BIN)/python -m pip install -e ".[dev,ui]"; \
	fi

.PHONY: lock
lock: ## Re-resolve uv.lock after editing dependencies in pyproject.toml
	uv lock

.PHONY: lock-check
lock-check: ## Fail if uv.lock has drifted from pyproject.toml
	uv lock --check

.PHONY: sample-data
sample-data: ## Regenerate the synthetic shelf images and planograms
	$(BIN)/python scripts/generate_sample_data.py --force

.PHONY: run
run: ## Start the API with autoreload on :8000
	$(BIN)/python -m smart_shelf --reload

.PHONY: ui
ui: ## Start the Streamlit front end on :8501
	$(BIN)/python -m smart_shelf.ui

.PHONY: test
test: ## Run the whole test suite
	$(BIN)/python -m pytest

.PHONY: test-unit
test-unit: ## Run only the fast unit tests
	$(BIN)/python -m pytest tests/unit -q

.PHONY: test-api
test-api: ## Run only the HTTP endpoint tests
	$(BIN)/python -m pytest tests/api -q

.PHONY: test-ui
test-ui: ## Run only the Streamlit front-end tests
	$(BIN)/python -m pytest tests/ui -q

.PHONY: cov
cov: ## Run tests with a coverage report
	$(BIN)/python -m pytest --cov=smart_shelf --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

.PHONY: fmt
fmt: ## Autofix lint findings and format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

.PHONY: typecheck
typecheck: ## Static type analysis
	$(BIN)/mypy --config-file mypy.ini

.PHONY: check
check: lock-check lint typecheck test ## Everything CI runs

.PHONY: docker-up
docker-up: ## Build and start the full stack
	docker compose up --build

.PHONY: docker-down
docker-down: ## Stop the stack and drop volumes
	docker compose down -v

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml junit.xml \
	       build dist src/*.egg-info
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
