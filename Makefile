.PHONY: clean clean_cache help

# Default values (can be overridden by CLI arguments)
DYNAMIC_SETTING ?= 0.6_0.8_1.0
DATASET ?= movie
QUERY_ID ?=
DRY_RUN ?= false

help:
	@echo "Usage:"
	@echo "  make clean_cache DATASET=movie DYNAMIC_SETTING=0.6_0.8_1.0 QUERY_ID=Q1 DRY_RUN=false"
	@echo ""
	@echo "Parameters:"
	@echo "  DATASET          - Dataset name (movie, ecomm, mmqa, medical)"
	@echo "  DYNAMIC_SETTING  - Dynamic setting (default: 0.6_0.8_1.0)"
	@echo "  QUERY_ID         - Query ID to delete (e.g., Q1, Q2, Q3a)"
	@echo "  DRY_RUN          - Show command only (true) or execute (false), default: true"
	@echo ""
	@echo "Examples:"
	@echo "  make clean_cache DATASET=movie QUERY_ID=Q1"
	@echo "  make clean_cache DATASET=movie QUERY_ID=Q1 DRY_RUN=false"
	@echo "  make clean_cache DATASET=medical QUERY_ID=Q3 DRY_RUN=false"
	@echo "  make clean_cache DATASET=mmqa QUERY_ID=Q6a DRY_RUN=false"

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
