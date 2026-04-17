# LiteDBX

A lightweight query engine for processing multi-modal data with semantic rule extraction, LLM-based inference, and intelligent query translation.

## Features

- **Semantic Query Processing**: Process complex semantic queries over multi-modal data using LLMs
- **Coreset Construction**: Intelligent sampling and coreset expansion for efficient labeling
- **Feature Space Generation**: Automatic feature extraction from text and image modalities
- **Feature Refinement**: LLM-driven feature space refinement with feedback loops
- **Label Propagation**: Semi-supervised learning with classifier-based label propagation
- **Query Translation**: Convert semantic queries to interpretable rule-based queries
- **Objective & Subjective Error Estimation**: Theoretical bounds for query translation quality
- **Multi-Modal Support**: Handle text, images, and structured tabular data

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- [vLLM](https://github.com/vllm-project/vllm) - For hosting local models (requires separate environment)

### Hardware Configuration

We host local models with vLLM on 4× NVIDIA GeForce RTX 3090 (24 GiB each)
- Supports simultaneous 30B LLM and 30B VLM hosting in FP8 (text/image multi-modal)

### Installation

#### 1. Main Project Installation

```bash
# Install dependencies using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

**Note**: The `uv sync` command does **not** include vLLM dependencies. You need to set up a separate environment for vLLM.

#### 2. vLLM Environment Setup

For hosting local models, you need to create a separate environment with vLLM:

```bash
# Create a dedicated venv for vLLM (recommended location: ~/venv/vllm)
python -m venv ~/venv/vllm
source ~/venv/vllm/bin/activate

# Install vLLM
pip install vllm
```

The [`servem.sh`](servem.sh) script automatically activates this venv at line 198. If your vLLM is installed in a different location, modify line 198 in [`servem.sh`](servem.sh:198) to point to your environment.

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
DASHSCOPE_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1/
```

### Locally-Hosted Models

For running LLMs locally, use the provided [`servem.sh`](servem.sh) script to serve models with vLLM:

#### Available Models

The script supports pre-configured models:

| Model | Type | Default Port | GPUs | Max Length |
|-------|------|--------------|------|------------|
| `llama3-8b` | Text | 8000 | 0 | 8192 |
| `qwen3-4b` | Text | 8001 | 1 | 32768 |
| `qwen3-30b-fp8` | Text | 8004 | 0,1 | 32768 |
| `qwen3-vl-2b` | Vision | 8006 | 2,3 | 32768 |
| `qwen3-vl-4b` | Vision | 8007 | 2,3 | 32768 |
| `qwen3-vl-8b` | Vision | 8002 | 2,3 | 32768 |
| `llava-v1.6-7b` | Vision | 8003 | 2,3 | 32768 |
| `qwen3-vl-30b` | Vision | 8005 | 2,3 | 32768 |

#### Usage

```bash
# List all available models
./servem.sh

# Start a specific model (e.g., Qwen3 4B)
./servem.sh qwen3-4b

# The server will start on the configured port
# Access the API at: http://localhost:<PORT>/v1
```

#### Model Configuration

Each model is pre-configured with the following settings:
- **Tensor Parallelism (TP)**: Automatically enabled for multi-GPU models
- **GPU Memory Utilization**: Configured per model (0.8-0.9)
- **Port Management**: Auto-increments if default port is in use
- **Virtual Environment**: Automatically activates `~/venv/vllm` (configurable at [line 198](servem.sh:198))

To add new models, edit the configuration arrays in [`servem.sh`](servem.sh:75-139).

### Data Setup

This project uses datasets from [SemBench](https://sembench.ngrok.io/). To set up the data:

1. **Download the data directory**:
   ```bash
   # Place the 'data' directory under ./litedbx/
   # Your directory structure should look like:
   # litedbx/
   #   ├── data/
   #   │   ├── medical/
   #   │   │   ├── data.csv
   #   │   │   └── ground_truth/
   #   │   └── ...
   #   └── ...
   ```

2. **Verify data setup**:
   ```bash
   # Check that the data directory exists
   ls -la data/

   # You should see dataset directories like:
   # medical/ movie/ ecomm/ animals/ mmqa/
   ```

## Usage

### Basic Usage

The query pipeline consists of three main steps:

1. **Define your workload** in `workloads/{dataset}.py` (e.g., `workloads/medical.py`)
2. **Configure the query parameters** in `workloads/config.yaml`
3. **Execute queries** using the engine

### Example Query Pipeline

```python
from time import time
import logging
import sys
from workloads import medical
from ldb_engine import LdbEngine
import asyncio

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
    queries = ["Q1"]

    # Build the workload
    workload = medical.get_workload(queries=queries)

    # Create the query engine
    ldb_engine = LdbEngine(workload)

    # Execute queries
    start = time()
    asyncio.run(ldb_engine.execute(debug=True))
    end = time()
    logger.info(f"Total execution time: {end - start} seconds")
```

### Running the Demo

```bash
python main.py
```

## Execution Phases

The `LdbEngine.execute()` method orchestrates the following phases:

1. **Preprocessing (Phase 1)**: Apply static filters (Σ) to retrieve sigma-satisfied data
2. **Coreset Construction (Phase 2)**: 
   - Initialize feature space with human labels
   - Materialize features for unlabeled data
   - Expand coreset using k-NN based selection
3. **Schema Selection & Query Translation (Phase 3)**:
   - Rank and trim feature space according to selection budget
   - Select optimal schema and translate queries

## Workload Configuration

Configure your queries in the `workloads/` directory. Each dataset should have its own workload module (e.g., `medical.py`).

### Configuration File (`workloads/config.yaml`)

```yaml
random_seed: 42
b_lab: 50          # Number of human labels to acquire
b_se: 5            # External feature selection budget
b_rew: 5           # Query rewriting (disjunction) budget
b_fs: 10           # Feature space generation budget
k_neighbors: 5     # Number of neighbors for coreset expansion
loo_step: 10       # Step size for leave-one-out validation
delta: 0.05        # Confidence level for error estimation
```

### Defining a Workload

```python
from pathlib import Path
from typing import Optional
import yaml
from data_structure import Predicate, SemPredicate, SemCQ
from .ldb_workload import LdbWorkload

DATASET_PATH = Path(__file__).parent.parent / "data/medical"
CURRENT_DIR = Path(__file__).parent

# Define semantic query
Q1 = SemCQ(
    selected=["patient_id"],
    Sigma=[Predicate("symptoms", "!=", "")],  # Static filter
    Ps=[
        SemPredicate(
            field="symptoms",
            modality="Text",
            succ_cond="The patient has an allergy",
            prompt=(
                "You are a medical expert. "
                "Please determine if the given symptom indicate "
                "that the patient has an allergy. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
            ))
    ]
)

SEM_QUERIES = {
    "Q1": Q1,
    # Add more queries...
}

def get_workload(queries: list[str], config: Optional[dict] = None) -> LdbWorkload:
    sem_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in medical dataset."
        sem_queries[q] = SEM_QUERIES[q]

    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)

    return LdbWorkload(
        data_dir=str(DATASET_PATH),
        scenario="medical",
        queries=sem_queries,
        config=config
    )
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `random_seed` | Random seed for reproducibility | 42 |
| `b_lab` | Number of human labels to acquire initially | 50 |
| `b_se` | Budget for selecting external features | 5 |
| `b_rew` | Budget for query rewriting disjunctions | 5 |
| `b_fs` | Budget for feature space generation | 10 |
| `k_neighbors` | Number of neighbors for coreset expansion | 5 |
| `loo_step` | Step size for leave-one-out validation | 10 |
| `delta` | Confidence parameter (1 - delta) for error bounds | 0.05 |

## Project Structure

```
litedbx/
├── ldb_engine.py          # Main query engine orchestration
├── main.py                # Entry point for running queries
├── servem.sh              # vLLM model serving script
├── workloads/             # Dataset-specific workloads
│   ├── config.yaml           # Configuration parameters
│   ├── ldb_workload.py       # LdbWorkload class
│   ├── medical.py            # Medical dataset queries
│   ├── movie.py              # Movie dataset queries
│   ├── ecomm.py              # E-commerce dataset queries
│   ├── animals.py            # Wildlife dataset queries
│   ├── mmqa.py               # MMQA dataset queries
│   ├── feature_utils.py      # Feature extraction utilities
│   └── workload_utils.py     # Workload helper functions
├── common/                # Common utilities
│   ├── coreset_selector.py   # Coreset selection algorithms
│   └── utils.py              # Feature encoding, classifiers, evaluation
├── data_structure/         # Data structures
│   ├── sem_query.py          # Semantic query (SemCQ, SemPredicate)
│   ├── ldb_data.py           # LdbData wrapper
│   ├── ldb_data_manager.py   # Data management
│   └── llm_resp_templates.py # LLM response templates
├── llm/                   # LLM integration
│   ├── config.yaml           # LLM configuration (remote/local)
│   ├── ldb_llm_client.py     # LLM API client
│   └── prompts.py            # LLM prompt templates
├── data/                  # Dataset directory
│   ├── medical/              # Medical dataset
│   ├── movie/                # Movie dataset
│   ├── ecomm_sf_2000/        # E-commerce dataset (scale factor 2000)
│   ├── animals/              # Wildlife dataset
│   └── mmqa/                 # MMQA dataset
├── .data_ckpt/            # Cached checkpoints
│   ├── medical/
│   ├── movie/
│   ├── ecomm/
│   └── ...
├── files/                 # External reference files
│   └── ...
└── litedbx_full.pdf       # Full-version paper (with appendix)
```

## Experimental Setup

### Models

**Remote (Cloud) Models:**
- Default LLM: `Qwen3-235B-A22B`
- Default VLM: `Qwen3-VL-235B-A22B-Instruct`

**Local Models (hosted via vLLM):**
- Default LLM: `Qwen3-30B-A3B-Instruct-2507-FP8`
- Default VLM: `Qwen3-VL-30B-A3B-Instruct-FP8`

### Benchmark

LiteDBX is evaluated on [SemBench](https://sembench.ngrok.io/), a benchmark for semantic query processing over multi-modal data.

**Datasets and Scale Factors:**

| Dataset | Scale Factor | Note |
|---------|--------------|------|
| Movie | 2000 | |
| Wildlife | 200 | |
| E-Commerce | 2000 | SemBench uses 500; LiteDBX uses 2000 |
| Medical | 11112 | |
| MMQA | 200 | |

### Query Mapping

| SemBench | LiteDBX |
|---------|---------|
| Movie.Q1, Q2 | Movie.Q1, Q2 |
| Wildlife.Q7 | Wildlife.Q1 |
| E-Commerce.Q1, Q2, Q13 | E-Commerce.Q1, Q2, Q3 |
| MMQA.Q3a, Q3f, Q6a, Q6b, Q6c | MMQA.Q1, Q2, Q3, Q4, Q5 |
| Medical.Q1, Q3, Q8, Q9 | Medical.Q1, Q2, Q3, Q4 |

### Evaluation Notes

1. **Aggregation Queries**: Excluded from evaluation as they require human-provided annotations (e.g., `SUM`, `COUNT`).

2. **LIMIT Clauses**: Removed from queries before execution (LIMIT optimization not supported).
