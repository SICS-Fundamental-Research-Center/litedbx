# LiteDBX

A lightweight query engine for multi-modal data with LLM-based feature extraction and query translation, with provable error certificates.

## Features

- **Semantic Query Processing**: run semantic queries (SemCQ) over text, image, and structured data
- **Coreset Construction**: label-efficient sampling with k-NN coreset expansion
- **Feature Space Generation & Refinement**: LLM-driven attribute extraction with feedback loops
- **Label Propagation**: classifier-based semi-supervised propagation to unlabeled data
- **Query Translation**: semantic predicates → interpretable bounded-UCQ rules
- **Error Certificates**: objective and subjective bounds on translation quality, including dynamic σ-pool certificates under data updates (with carryover for drained streams)
- **Incremental Maintenance**: selective refresh — reuse or refresh per query, guided by the certificate
- **Dynamic Data Streams**: update-stream execution and drifted-query reuse (`query_drift.py`)

## Setup

### Prerequisites

- Python 3.12+ and [uv](https://github.com/astral-sh/uv)
- [vLLM](https://github.com/vllm-project/vllm) (separate environment) for local model hosting

### Hardware

Local models are hosted with vLLM on 4× NVIDIA GeForce RTX 3090 (24 GiB each). Each modality has a dedicated front: text queries are served by `qwen3-30b-fp8` (:8004), image queries by `qwen3-vl-30b` (:8005), and audio queries by `qwen3-omni` (:8008 bf16 / :8009 AWQ).

### Installation

```bash
# Main environment (does not include vLLM)
uv sync && source .venv/bin/activate

# Dedicated vLLM environment (servem.sh activates it automatically)
python -m venv ~/venv/vllm
source ~/venv/vllm/bin/activate && pip install vllm
```

### Environment Configuration

```bash
mv .env.example .env   # then fill in your keys
```

| Variable | Purpose |
|----------|---------|
| `BLSC_API_KEY` / `BLSC_ENDPOINT` | BLSC gateway (remote models) |
| `DASHSCOPE_API_KEY` / `DASHSCOPE_ENDPOINT` | DashScope (remote models) |

## Serving Local Models

```bash
./servem.sh                     # list preconfigured models
./servem.sh start qwen3-30b-fp8 # serve a model (tmux); API at http://localhost:<PORT>/v1
./servem.sh stop qwen3-30b-fp8  # stop the session and clear state
./servem.sh status [MODEL]      # show session status
./servem.sh help                # full instructions
```

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
| `qwen3-omni-30b` | Omni (T+V+A) | 8008 | 0-3 | 32768 |
| `qwen3-omni-30b-awq` | Omni (T+V+A) | 8009 | 2,3 | 32768 |

Tensor parallelism is enabled automatically for multi-GPU models; GPU memory utilization is configured per model and ports auto-increment if busy. To add models, edit the configuration arrays at the top of [`servem.sh`](servem.sh).

## Data Setup

Download the [SemBench](https://sembench.ngrok.io/) `data/` directory into the repo root. Expected layout (plus scale-factor variants such as `movie_sf_16000`, `movie_sf_2000`, `mmqa_sf_25`):

```
data/
├── medical/    
├── movie/                  # full 1.54M sync; official-scale universe at movie_sf_2000/
├── movie_sf_2000/          # SemBench official scale-2000 universe (data_full + GT)
├── ecomm_sf_2000/    
├── animals/    
├── cars/                   # official scale sf_19672 (GT + per-branch rowsets via build_gt.py)
└── mmqa/
```

## Usage

### CLI (experiment harness)

```bash
uv run main.py --ls-configs                        # list experiment configs
uv run main.py --config configs/default.yaml       # run a config (repeat --config for several)
uv run main.py --config <cfg> --cold --certificate # no cache reads/writes + error certificates
```

Configs live under `exp/` as `<group>/<file>.yaml` task lists: each task names models, a `config_override` block (task-level knobs; the global `workloads/config.yaml` is shared live state), and workload/query objects — see `exp/configs/default.yaml`. Experiment directories under `exp/` are self-documenting: each carries its spec (`*_experiment.md`) and a script-generated `_summary.md`.

### Python API

```python
import asyncio, logging
from workloads.scenarios import medical
from ldb_engine import LdbEngine

logging.basicConfig(level=logging.INFO)

workload = medical.get_workload(queries=["Q1"])   # loads workloads/config.yaml
asyncio.run(LdbEngine(workload).execute(debug=True))
```

### Execution Phases

1. **Preprocessing (Phase 1)** — static filters (Σ) retrieve sigma-satisfied data
2. **Coreset Construction (Phase 2)** — seed with human labels, materialize features for unlabeled data, expand the coreset via k-NN selection
3. **Schema Selection & Query Translation (Phase 3)** — rank and trim the feature space under the selection budget, then rewrite the semantic predicate into rules

## Configuration

Global defaults in [`workloads/config.yaml`](workloads/config.yaml):

```yaml
random_seed: 42
b_lab: 20            # human labels acquired initially
b_se: 5              # external feature selection budget
b_rew: 5             # query rewriting (disjunction) budget
b_fs: 10             # feature space generation budget
k_neighbors: 5       # neighbors for coreset expansion
loo_step: 1          # leave-one-out validation step
delta: 0.2           # bounds hold with probability ≥ 1 − delta
dynamic_setting: [1.0]  # dynamic update plan (growth fractions)
```

Feature toggles (all `True` by default): `enable_hitl`, `enable_conf_struct`, `enable_conf_pred`, `enable_enrich`, `enable_rewrite`, `enable_subj`, `enable_obj`, `enable_coreset_expansion`.

Experiments override these per task via `config_override` in the experiment config — never by editing the global file.

### Defining a Workload

Each dataset is a module in `workloads/scenarios/`:

```python
from data_structure import Predicate, SemPredicate, SemCQ

Q1 = SemCQ(
    selected=["patient_id"],
    Sigma=[Predicate("symptoms", "!=", "")],       # static filter
    Ps=[SemPredicate(
        field="symptoms", modality="Text",
        succ_cond="The patient has an allergy",
        prompt="... judge whether the symptoms indicate an allergy ...",
    )],
)
SEM_QUERIES = {"Q1": Q1}

def get_workload(queries, config=None): ...        # -> LdbWorkload
```

## Project Structure

```
litedbx/
├── main.py                     # CLI entrance (experiment configs)
├── ldb_engine.py               # engine orchestration
├── query_drift.py              # reuse-aware execution of drifted query sequences
├── servem.sh                   # vLLM model serving
├── workloads/
│   ├── config.yaml             # global configuration (live shared state)
│   ├── config_schema.py        # config validation
│   ├── ldb_workload.py         # engine-facing workload facade
│   ├── registry.py             # workload registry
│   ├── utils.py                # encoding, classifiers, rules, losses
│   ├── core/                   # feature_pipeline, semantic_features, feature_selection,
│   │                           # rewrite_candidates, coreset_maintainer,
│   │                           # query_execution (rewrite, errors, incremental), preprocessing, reporting
│   └── scenarios/              # one package per dataset: movie, animals, ecomm, medical, mmqa, cars
├── data_structure/             # sem_query, ldb_data(_manager), coreset, data_stream,
│                               # sigma_satisfied_data, annotation_sampling, llm_resp_templates
├── llm/                        # config.yaml (remote/local model routing), ldb_llm_client, prompts
├── exp/                        # experiment campaigns: <NN>_<name>/ specs, configs, results, _summary.md
├── paper/                      # paper sources (main + appendix)
├── data/                       # datasets (see Data Setup)
└── litedbx_full.pdf            # paper with appendix
```

## Experimental Setup

### Models

| Regime | Text | Vision | Audio |
|--------|------|--------|-------|
| Remote (gateway) | `Qwen3.6-Plus` | `Qwen3.6-Plus` | `Qwen3.8-Omni-Flash` |
| Local (vLLM) | `Qwen3-30B-A3B-Instruct-2507-FP8` (:8004) | `Qwen3-VL-30B-A3B-Instruct-FP8` (:8005) | `Qwen3-Omni-30B-A3B` (:8008 bf16 / :8009 AWQ) |

Routing is configured in [`llm/config.yaml`](llm/config.yaml): every modality
is served by its dedicated front — text :8004, vision :8005, audio omni
:8008/:8009. (Some Nov 21–22 experiment runs routed Text/Image through the
omni @8009 to fit tri-modality queries on 4 GPUs; see those experiments'
reports — `revision/exp/R1.D5.1/` — for the disclosed deviation.)

### Benchmark

Evaluated on [SemBench](https://sembench.ngrok.io/):

| Dataset | Scale Factor | Note |
|---------|--------------|------|
| Movie | 2000 | official universe at `data/movie_sf_2000/` |
| Wildlife | 200 | |
| E-Commerce | 2000 | SemBench uses 500; LiteDBX uses 2000 |
| Medical | 11112 | |
| MMQA | 200 | two universes: `mmqa/movie`, `mmqa/airport` |
| Cars | 19672 | SemBench default 20,000 (seed 42) yields 19,672 rows |

### Query Mapping

41 queries in scope. SemBench numbering is authoritative; the
paper's old per-dataset renumbering is retired.

| SemBench | LiteDBX (repo qid) |
|----------|---------|
| Movie.Q1–Q4, Q8 | movie.Q1–Q4, Q8 (Q3 count, Q4 ratio, Q8 group-count added) |
| Wildlife.Q1–Q10 | animals.Q1–Q10 (Q7 pre-existing; +9 new) |
| E-Commerce.Q1, Q2, Q3 | ecomm.Q1, Q2, Q13 |
| MMQA.Q1–Q5 | mmqa.Q3a, Q3f, Q6a, Q6b, Q6c |
| Medical.Q1–Q9 | medical.Q1–Q9 (+5 new: Q2 audio, Q4 AVG, Q5 count, Q6/Q7 multimodal) |
| Cars.Q1–Q9 | cars.Q1–Q9 (new dataset) |

### Evaluation Notes

1. **Aggregation queries are in scope** since the domain expansion: scalar/group aggregates are
   evaluated by relative error and reported as `quality = 1/(1+rel_err) ∈ [0,1]` (non-F1); rowset
   filters use F1, argmax queries binary correctness.
2. **LIMIT clauses** are removed before execution (LIMIT optimization not supported).
