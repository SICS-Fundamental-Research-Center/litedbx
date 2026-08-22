# pylint: disable=missing-function-docstring,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,invalid-name
"""Query-drift patch: reuse-aware execution of drifted query sequences.

Independent entrance alongside main.py (no engine file is touched). A drift
sequence is a list of query names (len >= 2); the FIRST query is compiled by
the normal SEQR pipeline in-process, then every subsequent query is answered
under one of four modes:

- align:     Preprocessing.query_router(new, first) decides semantic
             alignment; aligned -> apply the first query's rewrite directly,
             otherwise full SEQR.
- selective: as align, but a failed alignment first asks whether the first
             query's materialized schema can support the new query; if yes,
             an LLM regenerates the UCQ over that schema (bypassing SEQR);
             otherwise full SEQR.
- reuse:     always apply the first query's rewrite directly.
- rerun:     always full SEQR.

Bypass evaluation mirrors the engine: rules are applied with
workloads.utils.apply_rules over encode_features(df) of the persisted
Sigma-satisfied view, and F1 follows data_structure.sigma_satisfied_data
semantics (ground_truth/<q>.csv[selected] as key tuples). Patch state
(rewrite UCQ + materialized view + metrics) is cached under
.data_ckpt/<exp_group>/mmqa/.../<q>/drift_state.json; the CSV round-trip
may shift dtypes vs the in-memory run (disclosed).

The serving stack (text + VL for Q6c) must already be up. Usage::

    uv run query_drift.py --mode selective --group recombine --seed 42
    uv run query_drift.py --mode align --queries Q3f Q3a
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import yaml
from pydantic import BaseModel

from data_structure import LdbDataManager, SemCQ, SemPredicate
from data_structure.llm_resp_templates import BooleanFeatureResponse
from data_structure.sigma_satisfied_data import _quality_metrics
from ldb_engine import LdbEngine
from llm import LdbLLMClient
from workloads.core.preprocessing import Preprocessing
from workloads.ldb_workload import LdbWorkload, experiment_checkpoint_path
from workloads.scenarios import mmqa
from workloads.utils import apply_rules, encode_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("query_drift")

Modality = Literal["Text", "Image", "VectorText", "VectorImage"]

# ---------------------------------------------------------------------------
# Drift scenarios (MMQA). Paraphrase prompts are DRAFTS -- revise freely;
# ground truth is copied from the base query at runtime (same gold set).
# ---------------------------------------------------------------------------
PARAPHRASES = {
    "Q3a_p": ("Q3a", "The film falls into the comedy genre",
              "Please determine whether the given text suggests that the "
              "film belongs to the comedy genre. Please JUST answer \"True\" "
              "if it does, and \"False\" otherwise. Do NOT provide any "
              "explanations."),
    "Q3c_p": ("Q3c", "The film belongs to the romance genre",
              "Please determine whether the given text suggests that the "
              "film belongs to the romance genre. Please JUST answer \"True\" "
              "if it does, and \"False\" otherwise. Do NOT provide any "
              "explanations."),
    "Q3f_p": ("Q3f", "The film is categorized as a romantic comedy",
              "Please determine whether the given text suggests that the "
              "film is categorized as a romantic comedy. Please JUST answer "
              "\"True\" if it does, and \"False\" otherwise. Do NOT provide "
              "any explanations."),
    "Q3g_p": ("Q3g", "The film can be classified as a biographical comedy",
              "Please determine whether the given text suggests that the "
              "film can be classified as a biographical comedy. Please JUST "
              "answer \"True\" if it does, and \"False\" otherwise. Do NOT "
              "provide any explanations."),
    "Q6a_p": ("Q6a", "The airline operates flights with Frankfurt as a destination",
              "Please determine whether the given text indicates that the "
              "airline operates flights with Frankfurt as a destination. "
              "Please JUST answer \"True\" if it does, and \"False\" "
              "otherwise. Do NOT provide any explanations."),
    "Q6b_p": ("Q6b", "The airline serves destinations within Germany",
              "Please determine whether the given text indicates that the "
              "airline serves destinations within Germany. Please JUST "
              "answer \"True\" if it does, and \"False\" otherwise. Do NOT "
              "provide any explanations."),
    "Q6c_p": ("Q6c", "The airline flies to destinations in Europe",
              "Please determine whether the given text indicates that the "
              "airline flies to destinations in Europe. Please JUST answer "
              "\"True\" if it does, and \"False\" otherwise. Do NOT provide "
              "any explanations."),
}

ALL_QUERIES = dict(mmqa.SEM_QUERIES)
for _p, (_base, _succ, _prompt) in PARAPHRASES.items():
    _baseq = mmqa.SEM_QUERIES[_base]
    ALL_QUERIES[_p] = SemCQ(
        selected=_baseq.selected,
        Sigma=list(_baseq.Sigma),
        Ps=[SemPredicate(
            field=_baseq.Ps[0].field,
            modality=cast(Modality, _baseq.Ps[0].modality),
            succ_cond=_succ,
            prompt=_prompt,
        )],
    )

# "Q3b" in the user's construction is a typo for Q3c (mmqa.py has no Q3b).
DRIFT_GROUPS = {
    "paraphrase": [[q, f"{q}_p"] for q in
                   ["Q3a", "Q3c", "Q3f", "Q3g", "Q6a", "Q6b", "Q6c"]],
    "recombine": [["Q3f", "Q3a"], ["Q3f", "Q3c"], ["Q3g", "Q3a"]],
    "unseen": [["Q3a", "Q3f"], ["Q3c", "Q3f"], ["Q3a", "Q3g"]],
    "revision": [[a, b] for a in ("Q6a", "Q6b", "Q6c")
                 for b in ("Q6a", "Q6b", "Q6c") if a != b],
}


def data_dir_of(q: str) -> Path:
    base = PARAPHRASES[q][0] if q in PARAPHRASES else q
    return mmqa.DATA_MAP[base]


def ensure_ground_truth(queries: list[str]) -> None:
    """Copy base-query ground truth for paraphrases lacking their own GT."""
    for q in queries:
        if q in PARAPHRASES:
            gt = data_dir_of(q) / "ground_truth" / f"{q}.csv"
            if not gt.exists():
                src = data_dir_of(q) / "ground_truth" / f"{PARAPHRASES[q][0]}.csv"
                gt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, gt)
                logger.info("Copied ground truth %s -> %s", src, gt)


# ---------------------------------------------------------------------------
# LLM gates (patch-side; the router is the engine's own oracle, verbatim)
# ---------------------------------------------------------------------------
class RuleCondition(BaseModel):
    feature: str
    op: str  # "<=" or ">"
    threshold: float


class RuleSetResponse(BaseModel):
    rules: list[list[RuleCondition]]


def _router(client: LdbLLMClient) -> Preprocessing:
    # query_router only touches llm_client; the other deps are placeholders.
    return Preprocessing(
        llm_client=client, data_manager=cast(LdbDataManager, None), queries={},
        ckpt_path=Path("."), usage_statistics=[{}], enable_cache=False,
    )


def aligned_with_first(client: LdbLLMClient, q_new: str, q_first: str):
    return _router(client).query_router(
        ALL_QUERIES[q_new], ALL_QUERIES[q_first]
    )


def schema_supports(client: LdbLLMClient, q_new: str, columns: list[str]):
    prompt = (
        "You are a data-reuse expert. The previous query materialized the "
        "following columns (schema):\n"
        + "\n".join(f"- {c}" for c in columns)
        + f"\n\nNew query: {ALL_QUERIES[q_new].Ps[0].succ_cond}\n\n"
        "Can these columns ALONE support answering the new query (are all "
        "needed semantic attributes already materialized)? Return True if "
        "yes, False otherwise."
    )
    resp = cast(
        BooleanFeatureResponse,
        client.invoke(
            is_remote=True, modality="Text",
            prompt=prompt, response_model=BooleanFeatureResponse,
        ),
    )
    return resp.value


def regen_rules(client: LdbLLMClient, q_new: str, q_first: str,
                prev_rules: list, columns: list[str]):
    def sql(rs):
        return " OR ".join(
            "(" + " AND ".join(f"{f} {op} {t:.4f}"
                               for f, t, op in rule) + ")"
            for rule in rs
        ) or "(empty -- previous query selected no rules)"

    prompt = (
        "You are an expert in query rewriting. Given the previous query's "
        "UCQ over a fixed materialized schema, rewrite it for the new query "
        "using ONLY the listed columns.\n\n"
        f"Previous query: {ALL_QUERIES[q_first].Ps[0].succ_cond}\n"
        f"Previous UCQ: {sql(prev_rules)}\n\n"
        "Available columns:\n"
        + "\n".join(f"- {c}" for c in columns)
        + f"\n\nNew query: {ALL_QUERIES[q_new].Ps[0].succ_cond}\n\n"
        "Return the new UCQ: a list of conjunctions (OR-ed together); each "
        "condition is (feature, op, threshold) with op in {'<=', '>'}. "
        "Every feature MUST be one of the listed columns."
    )
    resp = cast(
        RuleSetResponse,
        client.invoke(
            is_remote=True, modality="Text",
            prompt=prompt, response_model=RuleSetResponse,
        ),
    )
    return [[(c.feature, c.threshold, c.op) for c in conj]
            for conj in resp.rules]


# ---------------------------------------------------------------------------
# Engine bridge: run one query through full SEQR and persist patch state
# ---------------------------------------------------------------------------
def _override(args) -> dict:
    return {
        "random_seed": args.seed,
        "b_lab": args.b_lab,
        "dynamic_setting": [args.fraction],
    }


def _ckpt_root(args) -> Path:
    return experiment_checkpoint_path(
        exp_group=args.exp_group, scenario="mmqa",
        exp_patch=_override(args), dynamic_setting=[args.fraction],
    )


def _state_path(args, q: str) -> Path:
    return _ckpt_root(args) / q / "drift_state.json"


def _usage_totals(payload: dict) -> dict:
    stats = payload["usage_statistics"]
    return {
        "cost_usd": sum(v["total_cost"] for v in stats.values()),
        "total_tokens": sum(v["total_tokens"] for v in stats.values()),
    }


async def run_seqr(q: str, args) -> dict:
    """Full SEQR run for one query; captures rewrite + view + metrics."""
    with open(mmqa.CURRENT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
    workload = LdbWorkload(
        data_dir=str(data_dir_of(q)), scenario="mmqa",
        queries={q: ALL_QUERIES[q]}, config=config,
    )
    workload.inject_exp_setting(exp_group=args.exp_group, exp_patch=_override(args))
    workload.set_cache_enabled(not args.cold)

    start = time.time()
    payload = await LdbEngine(workload).execute(
        debug=args.debug, certificate=False
    )
    elapsed = time.time() - start

    trace = payload["execution_trace"]
    best = trace[max(trace, key=int)][q]
    trans = {k: v for k, v in best["trans_eval"].items()
             if k != "retrieved_data"}
    state = {
        "q": q,
        "rules": best["rules"],
        "features": best["features"],
        "time_s": elapsed,
        "durations": payload["phase_durations"],
        **_usage_totals(payload),
        **trans,
    }
    sp = _state_path(args, q)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sigma = workload.data_manager.sigma_satisfied_data[0][q]["ldb_data"].df
    sigma.to_csv(sp.parent / "drift_sigma.csv", index=False)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    logger.info("SEQR run for %s done: F1=%.4f cost=$%.4f (%.1fs) -> %s",
                q, trans["f1"], state["cost_usd"], elapsed, sp)
    return state


async def load_or_run_first(q: str, args) -> dict:
    sp = _state_path(args, q)
    if sp.exists():
        with open(sp, encoding="utf-8") as f:
            state = json.load(f)
        if not (sp.parent / "drift_sigma.csv").exists():
            logger.warning("State for %s exists but sigma CSV is missing; "
                           "recompiling.", q)
        else:
            logger.info("Reusing cached drift state for %s (F1=%.4f).",
                        q, state["f1"])
            return state
    return await run_seqr(q, args)


# ---------------------------------------------------------------------------
# Bypass execution + evaluation (engine-mirrored, no SEQR)
# ---------------------------------------------------------------------------
def eval_rules(df: pd.DataFrame, rules: list, sem_q: SemCQ, q: str) -> dict:
    labels = apply_rules(rules, encode_features(df))
    sel = sem_q.selected
    gt_df = pd.read_csv(data_dir_of(q) / "ground_truth" / f"{q}.csv")[sel]
    gt = {tuple(row) for row in gt_df.values.tolist()}
    retrieved = {tuple(row) for row in df.loc[labels.astype(bool), sel]
                 .values.tolist()}

    print(gt)
    print(retrieved)
    tp, fp, fn = (len(gt & retrieved), len(retrieved - gt), len(gt - retrieved))
    return _quality_metrics(q, 0, tp, fp, fn, omega=len(df))


# ---------------------------------------------------------------------------
# Pair runner
# ---------------------------------------------------------------------------
async def run_pair(group: str, pair: list[str], args, client) -> list[dict]:
    first = pair[0]
    st0 = await load_or_run_first(first, args)
    rows = [{"group": group, "pair": pair, "mode": args.mode, "q": first,
             "action": "compile", "f1": st0["f1"], "precision": st0["precision"],
             "recall": st0["recall"], "time_s": st0["time_s"],
             "cost_usd": st0["cost_usd"], "aligned": None,
             "schema_support": None}]

    sigma = pd.read_csv(_state_path(args, first).parent / "drift_sigma.csv")
    prev_rules = st0["rules"]

    for q in pair[1:]:
        row = {"group": group, "pair": pair, "mode": args.mode,
               "q_first": first, "q": q, "aligned": None,
               "schema_support": None, "action": None, "f1": None,
               "time_s": 0.0, "cost_usd": 0.0}

        def _apply(label, rules, journal=False):
            row["action"] = label
            if journal:
                row["rules"] = rules
            row.update({k: v for k, v in eval_rules(
                sigma, rules, ALL_QUERIES[q], q).items()
                if not isinstance(v, set)})

        if args.mode == "rerun":
            s = await run_seqr(q, args)
            row.update(action="seqr", f1=s["f1"], precision=s["precision"],
                       recall=s["recall"], time_s=s["time_s"],
                       cost_usd=s["cost_usd"])
        elif args.mode == "reuse":
            _apply("reuse", prev_rules)
        else:
            start = time.time()
            aligned, usage = aligned_with_first(client, q, first)
            row["aligned"] = aligned
            row["time_s"] = time.time() - start
            row["cost_usd"] = usage["total_cost"]
            if aligned:
                logger.info("Query %s aligned with %s; reusing rules.", q, first)
                _apply("reuse", prev_rules)
            elif args.mode == "selective":
                start = time.time()
                supported = schema_supports(client, q, list(sigma.columns))
                row["schema_support"] = supported
                rules = None
                if supported:
                    logger.info("Schema supports %s; regenerating UCQ.", q)
                    rules = regen_rules(client, q, first, prev_rules,
                                        list(sigma.columns))
                    logger.info("Regenerated UCQ for %s: %s", q, rules)
                    if any(c[0] not in sigma.columns
                           for conj in rules for c in conj):
                        logger.warning("Regenerated UCQ for %s references "
                                       "unknown columns; falling back to "
                                       "SEQR.", q)
                        rules = None
                row["time_s"] += time.time() - start
                if rules is not None:
                    _apply("rewrite_sql", rules, journal=True)
                else:
                    s = await run_seqr(q, args)
                    logger.info("Falling back to SEQR for %s.", q)
                    row.update(action="seqr", f1=s["f1"],
                               precision=s["precision"],
                               recall=s["recall"], time_s=s["time_s"],
                               cost_usd=s["cost_usd"])
            else:
                s = await run_seqr(q, args)
                row.update(action="seqr", f1=s["f1"],
                           precision=s["precision"], recall=s["recall"],
                           time_s=s["time_s"], cost_usd=s["cost_usd"])
        row["pair"] = list(pair)
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query-drift patch entrance")
    parser.add_argument("--mode", required=True,
                        choices=["align", "selective", "reuse", "rerun"])
    parser.add_argument("--group", default="all",
                        choices=["all", *DRIFT_GROUPS])
    parser.add_argument("--queries", nargs="+",
                        help="Explicit drift sequence (len >= 2), e.g. "
                             "--queries Q3f Q3a; overrides --group.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b-lab", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--exp-group", default="query_drift")
    parser.add_argument("--cold", action="store_true",
                        help="Disable engine cache reads/writes for SEQR runs.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--out", default=None,
                        help="Journal jsonl path (default under "
                             "exp/21_query_drift/_results/).")
    return parser


async def drift_main(args) -> None:
    client = LdbLLMClient()
    if args.queries:
        if len(args.queries) < 2:
            raise SystemExit("--queries needs at least 2 names.")
        bad = [q for q in args.queries if q not in ALL_QUERIES]
        if bad:
            raise SystemExit(f"Unknown queries: {bad}")
        pairs = [("explicit", args.queries)]
    else:
        groups = ([g for g in DRIFT_GROUPS] if args.group == "all"
                  else [args.group])
        pairs = [(g, p) for g in groups for p in DRIFT_GROUPS[g]]
    ensure_ground_truth([q for _, p in pairs for q in p])

    out = Path(args.out) if args.out else (
        Path(__file__).parent / "exp" / "21_query_drift" / "_results"
        / f"drift_{args.mode}_seed{args.seed}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    for group, pair in pairs:
        rows = await run_pair(group, pair, args, client)
        with open(out, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        for row in rows:
            logger.info("[%s/%s] %s->%s %s: F1=%.4f $%.4f %.1fs",
                        group, args.mode, pair[0], row["q"], row["action"],
                        row["f1"] or 0.0, row["cost_usd"], row["time_s"])
    logger.info("Journal written to %s", out)


if __name__ == "__main__":
    asyncio.run(drift_main(build_parser().parse_args()))
