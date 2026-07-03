.PHONY: clean clean_cache code-stat help lint

# Default values (can be overridden by CLI arguments)
DYNAMIC_SETTING ?= 0.6_0.8_1.0
DATASET ?= movie
QUERY_ID ?=
DRY_RUN ?= false

help:
	@echo "Usage:"
	@echo "  make lint LINT_FILES=path/to/file.py"
	@echo "  make lint-fix LINT_FILES=path/to/file.py"
	@echo "  make clean_cache DATASET=movie DYNAMIC_SETTING=0.6_0.8_1.0 QUERY_ID=Q1 DRY_RUN=false"
	@echo "  make code-stat"
	@echo ""
	@echo "Linting:"
	@echo "  LINT_FILES       - Required Python file list for Ruff and Pylint"
	@echo "  lint             - Check only; does not modify files"
	@echo "  lint-fix         - Apply Ruff fixes/format, then run lint"
	@echo "  Commit hook      - uv run pre-commit run --files path/to/file.py"
	@echo ""
	@echo "Cache parameters:"
	@echo "  DATASET          - Dataset name (movie, ecomm, mmqa, medical)"
	@echo "  DYNAMIC_SETTING  - Dynamic setting (default: 0.6_0.8_1.0)"
	@echo "  QUERY_ID         - Query ID to delete (e.g., Q1, Q2, Q3a)"
	@echo "  DRY_RUN          - Show command only (true) or execute (false), default: true"
	@echo ""
	@echo "Code statistics:"
	@echo "  code-stat        - Count source lines with ../cloc/cloc, excluding third-party files/ and CSV/JSON data"
	@echo ""
	@echo "Examples:"
	@echo "  make lint LINT_FILES=main.py"
	@echo "  make lint-fix LINT_FILES=main.py"
	@echo "  make lint LINT_FILES='main.py exp/experiment_runner.py'"
	@echo "  uv run pre-commit run --files main.py exp/experiment_runner.py"
	@echo "  make clean_cache DATASET=movie QUERY_ID=Q1"
	@echo "  make clean_cache DATASET=movie QUERY_ID=Q1 DRY_RUN=false"
	@echo "  make clean_cache DATASET=medical QUERY_ID=Q3 DRY_RUN=false"
	@echo "  make clean_cache DATASET=mmqa QUERY_ID=Q6a DRY_RUN=false"

code-stat:
	../cloc/cloc . --exclude-dir=files --exclude-ext=csv,json,md

lint:
	@if [ -z "$(LINT_FILES)" ]; then \
		echo "Usage: make lint LINT_FILES=path/to/file.py"; \
		exit 2; \
	fi
	uv run ruff format $(LINT_FILES)
	uv run ruff check --fix $(LINT_FILES)
	PYTHONPATH=. uv run pylint $(LINT_FILES)

clean_cache:
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET parameter is required"; \
		exit 1; \
	fi
	@if [ -z "$(QUERY_ID)" ]; then \
		echo "Error: QUERY_ID parameter is required"; \
		exit 1; \
	fi
	@echo "Target path: .data_ckpt/$(DATASET)/$(DYNAMIC_SETTING)/$(QUERY_ID)"
	@if [ "$(DRY_RUN)" = "true" ]; then \
		echo "DRY RUN: Command that would be executed:"; \
		echo "  rm -rf .data_ckpt/$(DATASET)/$(DYNAMIC_SETTING)/$(QUERY_ID)"; \
		echo ""; \
		echo "To execute, run: make clean_cache DATASET=$(DATASET) DYNAMIC_SETTING=$(DYNAMIC_SETTING) QUERY_ID=$(QUERY_ID) DRY_RUN=false"; \
	else \
		echo "EXECUTING: Deleting .data_ckpt/$(DATASET)/$(DYNAMIC_SETTING)/$(QUERY_ID)"; \
		rm -rf .data_ckpt/$(DATASET)/$(DYNAMIC_SETTING)/$(QUERY_ID); \
		echo "Done!"; \
	fi
