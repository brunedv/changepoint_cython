SHELL := /bin/bash

REPORTS_DIR := notebooks
DOCUMENTATION_OUTPUT = $(REPORTS_DIR)/documentation
SHA := $(shell git rev-parse --short HEAD)
VERSION := $(shell tail VERSION | cut -c 2-)
SOURCE_DIR := pychangepoints
APIDOC_OPTIONS := -f -e -M -T -d 1 -H "API Documentation" -A "pychangepoints" -V $(VERSION) -R $(SHA)
COVERAGE_OPTIONS = --cov-branch --cov-config coverage/.coveragerc --cov-report term --cov-report term-missing

.PHONY: init setup markdown notebooks build doc tests coverage lint black-check black dist help

install:
	poetry install --all-extras

setup-dev: ## install dev dependencies
	poetry install --only dev

build: # build wheel python package
	poetry build

doc: ## setup-dev
	rm -rf docs/source/generated
	poetry run sphinx-apidoc $(APIDOC_OPTIONS) -o docs/source/generated/ $(SOURCE_DIR) $(SOURCE_DIR)/tests
	cd docs; poetry run make html
	mkdir -p $(DOCUMENTATION_OUTPUT)
	cp -r docs/build/html/* $(DOCUMENTATION_OUTPUT)

unit-tests:
	poetry run pytest tests/

format:  ## Autoformat project codebase with black and isort
	poetry run black $(SOURCE_DIR)
	poetry run isort $(SOURCE_DIR) --profile black

clean-doc: ## clean sphinx directory created
	rm -rf docs/build docs/source/generated
	rm -rf notebooks/documentation