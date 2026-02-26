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

### Locally-Hosted Models

For running LLMs locally, use the provided [`servem.sh`](servem.sh) script to serve models with vLLM:

#### Available Models

The script supports pre-configured models:

| Model | Type | Default Port | GPUs | Max Length |
|-------|------|--------------|------|------------|
| `llama3-8b` | Text | 8000 | 0 | 8192 |
| `qwen3-4b` | Text | 8001 | 1 | 32768 |
| `qwen3-30b` | Text | 8004 | 0,1 | 32768 |
| `qwen3-vl-8b` | Vision | 8002 | 2,3 | 32768 |
| `llava-v1.6-7b` | Vision | 8003 | 0,1 | 32768 |

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
- **Virtual Environment**: Automatically activates `~/venv/vllm`

To add new models, edit the configuration arrays in [`servem.sh`](servem.sh:75-117).

### Data Setup

This project uses datasets from [SemBench](https://github.com/SemBench/SemBench). To set up the data:

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
   # medical/
   # ```

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
    queries = ["Q1", "Q3", "Q8"]

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

1. **Preprocessing (Phase 1.1)**: Apply static filters (Σ) to retrieve sigma-satisfied data
2. **Coreset Initialization (Phase 2.1)**: Acquire human labels and initialize feature space
3. **Feature Materialization (Phase 2.2)**: Populate features for unlabeled data
4. **Coreset Expansion (Phase 2.3)**: Expand coreset using k-NN based selection
5. **Feature Generation (Phase 3.1)**: Generate candidate external features
6. **Schema Selection & Query Translation (Phase 3.2)**: Select optimal schema and translate queries

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

- `random_seed`: Random seed for reproducibility (default: 42)
- `b_lab`: Number of human labels to acquire initially (default: 50)
- `b_se`: Budget for selecting external features (default: 5)
- `b_rew`: Budget for query rewriting disjunctions (default: 5)
- `b_fs`: Budget for feature space generation (default: 10)
- `k_neighbors`: Number of neighbors for coreset expansion (default: 5)
- `loo_step`: Step size for leave-one-out validation (default: 10)
- `delta`: Confidence parameter (1 - delta) for error bounds (default: 0.05)

## Project Structure

```
litedbx/
├── ldb_engine.py          # Main query engine orchestration
├── main.py                # Entry point for running queries
├── common/                # Common utilities
│   ├── coreset_selector.py   # Coreset selection algorithms
│   └── utils.py              # Feature encoding, classifiers, evaluation
├── data_structure/         # Data structures
│   ├── sem_query.py          # Semantic query (SemCQ, SemPredicate)
│   ├── ldb_data.py           # LdbData wrapper
│   └── llm_resp_templates.py # LLM response templates
├── llm/                   # LLM integration
│   ├── ldb_llm_client.py     # LLM API client
│   └── prompts.py            # LLM prompt templates
├── workloads/             # Dataset-specific workloads
│   ├── config.yaml           # Configuration parameters
│   ├── ldb_workload.py       # LdbWorkload class
│   └── medical.py            # Medical dataset queries
├── data/                  # Dataset directory
│   └── medical/              # Medical dataset
│       ├── data.csv
│       └── ground_truth/
└── .ckpt/                 # Cached checkpoints
    └── medical/              # Medical dataset checkpoints
```
