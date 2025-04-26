PYTHON_VERSION := python3.12
DEPLOY_DIR := .deploy
PYTEST_ARGS := --doctest-modules --cov
PYTEST_REPORT_ARGS :=  --cov-report=xml:coverage.xml

.PHONY: install
install:
	@poetry env use $(PYTHON_VERSION)
	@poetry install
	@poetry run pre-commit install

.PHONY: run
run:
	@cd $(DEPLOY_DIR) && docker-compose up --build

.PHONY: lint.mypy
lint.mypy:
	@poetry run mypy

.PHONY: test.pytest
test.pytest:
	@pytest $(PYTEST_ARGS) -- tests

.PHONY: test.coverage
test.coverage:
	@pytest $(PYTEST_ARGS) $(PYTEST_REPORT_ARGS) -- tests

.PHONY: pre-commit-all
pre-commit-all:
	@pre-commit run --all-files

.PHONY: align_code
align_code:
	@ruff check . --fix && ruff format .
