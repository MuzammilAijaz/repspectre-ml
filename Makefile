.PHONY: help lint test db-diagram run-visualizer

DATABASE_FILE ?= ./data/sensor-database_06-09-2026_prototype_proper.db

# Default target when you just type 'make'
help:
	@echo "Available commands:"
	@echo "  make lint       	 - Run Ruff and Basedpyright on entire project"
	@echo "  make test       	 - Run pytest with coverage"
	@echo "  make db-diagram 	 - Generate database ER graph using eralchemy2"
	@echo "  make run-visualizer - Run data analysis/visualizer tool"

lint:
	-uv run ruff check .
	-uv run basedpyright

test:
	uv run pytest

db-to-csv:
	uv run src/lift_ml/data/prepare_dataset.py \
	--db "$(DATABASE_FILE)"

db-to-diagram:
	uv run eralchemy2 -i sqlite:///$(DATABASE_FILE) -o docs/schema.png
	@echo "Database diagram generated at docs/schema.png"

run-visualizer:
	uv run python tools/data-visualizer/main.py
