.PHONY: help install install-dev test test-unit test-integration lint format type-check clean build docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and core dependencies
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev,all]"

test: ## Run all tests
	pytest

test-unit: ## Run only unit tests
	pytest tests/unit/ -v

test-integration: ## Run only integration tests
	pytest tests/integration/ -v

test-cov: ## Run tests with coverage report
	pytest --cov=src/rag_agent --cov-report=html --cov-report=term

lint: ## Run linting checks
	flake8 src/ tests/
	isort --check-only --diff src/ tests/
	black --check src/ tests/

format: ## Format code with black and isort
	isort src/ tests/
	black src/ tests/

type-check: ## Run type checking with mypy
	mypy src/rag_agent

quality-check: lint type-check ## Run all code quality checks

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build the package
	python -m build

docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"

run-example: ## Run the basic usage example
	python examples/basic_usage.py

check-deps: ## Check for security vulnerabilities in dependencies
	pip-audit

setup-pre-commit: ## Setup pre-commit hooks
	pre-commit install