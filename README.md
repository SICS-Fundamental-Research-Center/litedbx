# LiteDBX

A lightweight query engine for processing multi-modal data with semantic rule extraction and LLM-based inference.

## Features

- **Semantic Rule Extraction**: Extract interpretable rules from unstructured data using LLMs
- **Query Rewriting**: Automatic query optimization and rewriting
- **Feature Enrichment**: Dynamic feature generation from multi-modal sources
- **Multi-Modal Support**: Handle text, images, and structured tabular data
- **UCQ/CQ Framework**: Union of Conjunctive Queries for complex reasoning

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

### Installation

```bash
# Install dependencies using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Environment Configuration

Configure your API keys by setting up the `.env` file:

```bash
# Rename the env file
mv .env.example .env

# Edit .env with your actual API keys
# Required variables:
# - BLSC_API_KEY: Your BLSC API key
# - DASHSCOPE_API_KEY: Your DashScope API key
```

Example `.env` file:
```env
BLSC_API_KEY=your_api_key_here
BLSC_ENDPOINT=https://llmapi.blsc.cn/v1/
DASHSCOPE_API_KEY=your_dashscope_key_here
DASHSCOPE_ENDPOINT=thttps://dashscope.aliyuncs.com/compatible-mode/v1/
```

### Data Setup

This project uses datasets from [SemBench](https://github.com/SemBench/SemBench). To set up the data:

1. **Download the files directory**:
   ```bash
   # Download the files directory from SemBench
   # Visit: https://github.com/SemBench/SemBench/tree/main/files
   # Place the 'files' directory under ./litedbx/
   #
   # Your directory structure should look like:
   # litedbx/
   #   ├── files/
   #   │   ├── medical/
   #   │   └── ...
   #   └── ...
   ```

2. **Download multi-modal data** (images, etc.):
   - The raw multi-modal data is not included in the GitHub repository due to size
   - Download from: [Google Drive](https://drive.google.com/drive/folders/1pqf8DKFai16MR80Z7pcls5FgBbom-IJt?dmr=1&ec=wgc-drive-globalnav-goto)
   - Extract and place the multi-modal files in the corresponding `files/{dataset}/` directories

3. **Verify data setup**:
   ```bash
   # Check that the files directory exists
   ls -la files/

   # You should see dataset directories like:
   # medical/
   # ...
   ```

## Usage

### Basic Usage

The query pipeline consists of three main steps:

1. **Define your workload** in `workloads/{dataset}/` (e.g., `workloads/medical_workloads.py`)
2. **Configure the query engine** with feature enrichment and query rewrite budgets
3. **Execute queries** using the engine

### Example Query Pipeline

```python
from time import time
import asyncio
import logging
import sys
from workloads import medical_workloads

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Define which queries to run
    workloads = ["Q1"]

    # Build the query engine
    ldb_engine = medical_workloads.build_query_engine(
        workloads=workloads,
        feature_enrich_budget=3,      # Max feature enrichment iterations
        query_rewrite_budget=3,       # Max query rewrite iterations
    )

    # Execute queries
    start = time()
    asyncio.run(
        ldb_engine.apply(
            queries=workloads,
            enable_proxies=True  # Enable proxy services for external data
        )
    )
    end = time()
    logger.info(f"Total execution time: {end - start} seconds")
```

### Running the Demo

```bash
python main.py
```

## Workload Configuration

Configure your queries in the `workloads/` directory. Each dataset should have its own workload module (e.g., `medical_workloads.py`).

### Defining a Workload

```python
from ldb_engine import LDBEngine
from data_structures import UCQ, CQ

def q1():
    return UCQ(
        select_cols=["patient_id"],
        rules=[
            CQ(
                static_rules=[],
                sem_rules=[("symptoms", (
                    "You are a medical expert."
                    "Please determine if the following symptoms indicate an allergy."
                    "Please JUST answer \"True\" if they do, and \"False\" otherwise."
                    "Do NOT provide any explanations."))],
            ),
        ],
    )

WORKLOADS = {
    "Q1": q1(),
}

def build_query_engine(workloads, feature_enrich_budget=3, query_rewrite_budget=3):
    return LDBEngine(
        dataset_name="medical",
        workloads=_retrieve_workloads(workloads),
        feature_enrich_budget=feature_enrich_budget,
        query_rewrite_budget=query_rewrite_budget,
        external_keys=["image_path", "skin_image_id", "image_path_xray",
                       "xray_id", "symptoms", "symptom_id"]
    )
```

## Parameters

- `feature_enrich_budget`: Maximum number of feature enrichment iterations (default: 3)
- `query_rewrite_budget`: Maximum number of query rewrite iterations (default: 3)
- `enable_proxies`: Enable external proxy services for data retrieval (default: True)
- `external_keys`: List of external/modality columns to process

## Project Structure

- `ldb_engine.py`: Main query engine implementation
- `data_structures.py`: UCQ and CQ data structures
- `llm_client.py`: LLM API client
- `feature_gen.py`: Feature generation utilities
- `rule_filter.py`: Rule filtering and optimization
- `semantic_ops.py`: Semantic operations and embeddings
- `evaluation.py`: Evaluation metrics
- `prompts.py`: LLM prompt templates
- `workloads/`: Dataset-specific query definitions
- `main.py`: Example query pipeline
