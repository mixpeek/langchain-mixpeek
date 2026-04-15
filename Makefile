.PHONY: all format lint test tests integration_tests help

all: help

TEST_FILE ?= tests/

test tests:
	python -m pytest $(TEST_FILE) -q

integration_test integration_tests:
	python -m pytest tests/ -q -m integration

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

lint:
	ruff check src/ tests/
	mypy src/langchain_mixpeek/

build:
	rm -rf dist/
	python -m build

publish: build
	python -m twine upload --repository langchain-mixpeek dist/*

help:
	@echo "Available targets:"
	@echo "  test              - Run unit tests"
	@echo "  integration_test  - Run integration tests"
	@echo "  format            - Format code with ruff"
	@echo "  lint              - Lint + type check"
	@echo "  build             - Build wheel and sdist"
	@echo "  publish           - Build and upload to PyPI"
