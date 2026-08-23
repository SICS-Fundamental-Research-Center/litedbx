# pylint: disable=missing-function-docstring,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,invalid-name
"""Query-drift patch: reuse-aware execution of drifted query sequences.

Independent entrance alongside main.py (no engine file is touched). A drift
sequence is a list of query names (len >= 2); the FIRST query is compiled by
the normal SEQR pipeline in-process, then every subsequent query is answered
under one of four modes:

- align:     an LLM gate relates new vs first: same -> apply the first
             query's rewrite directly; contained (results(new) is a
             subset of results(first)) -> the prev UCQ's predicates are
             appended to the new query's Sigma at runtime and the query
             runs the normal SEQR pipeline over the shrunk scope;
             neither -> full SEQR.
- selective: as align, but on neither first ask whether the first
             query's materialized schema can support the new query; if yes,
             an LLM regenerates the UCQ over that schema (bypassing SEQR);
             otherwise full SEQR.
- reuse:     always apply the first query's rewrite directly.
- rerun:     always full SEQR.

Bypass evaluation is pure rule application: the state persists ONE
combined view (residual rows + the compile-time selected/discarded
annotation buckets, concatenated); retrieved = rule matches over
encode_features of that view -- annotated rows carry no materialized
features, so rules never fire on them and NO annotation label is ever
used in evaluation; GT = the FULL ground_truth/<q>.csv[selected] key
set; omega = len(view) = the compile's candidate_size. Patch state
(rewrite UCQ + view + metrics) is cached under
.data_ckpt/<exp_group>/mmqa/.../<q>/drift_state.json.

The text serving stack must already be up (Q6c's VectorText routes to
the text front). Usage::

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
from copy import deepcopy
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import yaml
from pydantic import BaseModel

from data_structure import Predicate, SemCQ, SemPredicate
from data_structure.llm_resp_templates import BooleanFeatureResponse
from data_structure.sigma_satisfied_data import _quality_metrics
from ldb_engine import LdbEngine
from llm import LdbLLMClient
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
    "Q3a_p": ("Q3a", "Pictures that fail to amuse lie outside the scope "
              "of this request",
              "Please determine whether the given text suggests that the "
              "film belongs to the comedy genre. Please JUST answer \"True\" "
              "if it does, and \"False\" otherwise. Do NOT provide any "
              "explanations."),
    "Q3c_p": ("Q3c", "An affair of the heart drives this picture's story",
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
                   ["Q3a", "Q3f", "Q3g", "Q6a", "Q6b", "Q6c"]],
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


class RelationResponse(BaseModel):
    relation: Literal["same", "contained", "neither"]


def aligned_with_first(
        client: LdbLLMClient, q_new: str, q_first: str
        ) -> tuple[str, float]:
    """Relate the new query to the first: same / contained / neither."""
    prompt = (
        "You are a query-reuse expert. Compare two queries:\n"
        f"A (new): {ALL_QUERIES[q_new].Ps[0].succ_cond}\n"
        f"B (previous): {ALL_QUERIES[q_first].Ps[0].succ_cond}\n"
        "Answer with exactly one word:\n"
        "- same: A and B ask for the same results (mere rewording).\n"
        "- contained: EVERY result of A is also a result of B (A's "
        "results are a subset of B's), e.g. A = 'destinations in "
        "Germany', B = 'destinations in Europe'.\n"
        "- neither: anything else, including B's results being a "
        "subset of A's."
    )
    client.reset_usage_statistics()
    resp = cast(
        RelationResponse,
        client.invoke(
            is_remote=True, modality="Text",
            prompt=prompt, response_model=RelationResponse,
        ),
    )
    cost = client.get_usage_statistics()["total_cost"]
    logger.info("LLM aligned %s vs %s: %s (cost=$%.4f)",
                q_new, q_first, resp.relation, cost)
    client.reset_usage_statistics()
    return resp.relation, cost


def schema_supports(
        client: LdbLLMClient, q_new: str, columns: list[str]
        ) -> tuple[bool, float]:
    prompt = (
        "You are a data-reuse expert. The previous query materialized the "
        "following columns (schema):\n"
        + "\n".join(f"- {c}" for c in columns)
        + f"\n\nNew query: {ALL_QUERIES[q_new].Ps[0].succ_cond}\n\n"
        "Can these columns ALONE support answering the new query (are all "
        "needed semantic attributes already materialized)? Return True if "
        "yes, False otherwise."
    )
    client.reset_usage_statistics()
    resp = cast(
        BooleanFeatureResponse,
        client.invoke(
            is_remote=True, modality="Text",
            prompt=prompt, response_model=BooleanFeatureResponse,
        ),
    )
    cost = client.get_usage_statistics()["total_cost"]
    logger.info("LLM schema support for %s: %s (cost=$%.4f)",
                q_new, resp.value, cost)
    client.reset_usage_statistics()
    return resp.value, cost


def regen_rules(client: LdbLLMClient, q_new: str, q_first: str,
                prev_rules: list, columns: list[str]
                ) -> tuple[list, float]:
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
    client.reset_usage_statistics()
    resp = cast(
        RuleSetResponse,
        client.invoke(
            is_remote=True, modality="Text",
            prompt=prompt, response_model=RuleSetResponse,
        ),
    )
    cost = client.get_usage_statistics()["total_cost"]
    logger.info("LLM regenerated UCQ for %s vs %s: %s (cost=$%.4f)",
                q_new, q_first, resp.rules, cost)
    client.reset_usage_statistics()
    rules = [[(c.feature, c.threshold, c.op) for c in conj]
             for conj in resp.rules]
    return rules, cost


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


async def run_seqr(q: str, args, sem: SemCQ | None = None,
                   materialize: bool = True,
                   reused_view: pd.DataFrame | None = None) -> dict:
    """Full SEQR run for one query; captures rewrite + view + metrics."""
    with open(mmqa.CURRENT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
    workload = LdbWorkload(
        data_dir=str(data_dir_of(q)), scenario="mmqa",
        queries={q: sem or ALL_QUERIES[q]}, config=config,
    )
    workload.inject_exp_setting(exp_group=args.exp_group,
                                exp_patch=_override(args))
    workload.set_cache_enabled(not args.cold)

    if reused_view is not None:
        workload.data_manager.complete_dataset.df = reused_view

    start = time.time()
    payload = await LdbEngine(workload).execute(
        debug=args.debug, certificate=False
    )
    elapsed = time.time() - start

    data_manager = workload.data_manager
    enriched_features = data_manager.enriched_features
    sigma_satisfied = data_manager.sigma_satisfied_data
    await sigma_satisfied.sync_annotation_features(
        q_name=q,
        enriched_features=enriched_features,
        llm_client=workload.llm_client,
    )

    trace = payload["execution_trace"]
    if not trace:
        raise RuntimeError(
            "execution_trace is empty; enable_rewrite/enable_enrich must "
            "be True for a SEQR run."
        )
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
    if materialize:  # only the first query of a pair persists its state
        sp = _state_path(args, q)
        sp.parent.mkdir(parents=True, exist_ok=True)
        record = workload.data_manager.sigma_satisfied_data[0][q]
        residual = record["ldb_data"].df
        view = pd.concat(
            [residual, record["selected_data"], record["discarded_data"]],
            ignore_index=True,
        )
        view.to_csv(sp.parent / "drift_view.csv", index=False)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=float)
        logger.info("SEQR run for %s done: F1=%.4f cost=$%.4f (%.1fs) -> %s",
                    q, trans["f1"], state["cost_usd"], elapsed, sp)
    else:
        logger.info("SEQR run for %s done: F1=%.4f cost=$%.4f (%.1fs)",
                    q, trans["f1"], state["cost_usd"], elapsed)
    return state


async def load_or_run_first(q: str, args) -> dict:
    sp = _state_path(args, q)
    if sp.exists():
        with open(sp, encoding="utf-8") as f:
            state = json.load(f)
        if not (sp.parent / "drift_view.csv").exists():
            logger.warning("State for %s exists but view CSV is missing; "
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
    """Bypass eval: pure rule matches vs the FULL ground truth."""
    labels = apply_rules(rules, encode_features(df))
    sel = sem_q.selected
    gt_df = pd.read_csv(data_dir_of(q) / "ground_truth" / f"{q}.csv")[sel]
    gt = {tuple(row) for row in gt_df.values.tolist()}
    retrieved = {tuple(row) for row in
                 df.loc[labels.astype(bool), sel].values.tolist()}

    tp, fp, fn = (len(gt & retrieved), len(retrieved - gt), len(gt - retrieved))

    logger.info(f"Ground truth for {q}: {gt}")
    logger.info(f"Retrieved for {q}: {retrieved}")
    logger.info(f"Evaluating {q}: TP={tp}, FP={fp}, FN={fn}, Omega={len(df)}")
    return _quality_metrics(q, 0, tp, fp, fn, omega=len(df))


# ---------------------------------------------------------------------------
# Pair runner
# ---------------------------------------------------------------------------
async def run_pair(group: str, pair: list[str], args, client) -> list[dict]:
    first = pair[0]
    if any(data_dir_of(q) != data_dir_of(first) for q in pair):
        raise SystemExit(f"Drift sequence spans datasets: {pair}")
    st0 = await load_or_run_first(first, args)
    rows = [{"group": group, "pair": pair, "mode": args.mode,
             "q_first": None, "q": first,
             "action": "compile", "f1": st0["f1"], "precision": st0["precision"],
             "recall": st0["recall"], "time_s": st0["time_s"],
             "cost_usd": st0["cost_usd"], "aligned": None,
             "schema_support": None}]

    view = pd.read_csv(_state_path(args, first).parent / "drift_view.csv")
    obj_cols = view.select_dtypes(include="object").columns
    view[obj_cols] = view[obj_cols].fillna("")
    columns = list(view.columns)
    prev_rules = st0["rules"]
    if not prev_rules:
        prev_rules = [[(c, 0.5, ">") for c in columns
                       if c.startswith("llm_label_")]]

    for q in pair[1:]:
        row = {"group": group, "pair": pair, "mode": args.mode,
               "q_first": first, "q": q, "aligned": None,
               "schema_support": None, "action": None, "f1": None,
               "time_s": 0.0, "cost_usd": 0.0}

        def _apply(label, rules, journal=False):
            row["action"] = label
            if journal:
                row["rules"] = rules
            t0 = time.time()
            m = eval_rules(view, rules, ALL_QUERIES[q], q)
            # bypass eval is pure local compute (no LLM) but not free —
            # measure it instead of journaling the 0.0 init value
            row["time_s"] += time.time() - t0
            row.update({k: v for k, v in m.items()
                        if not isinstance(v, set)})

        async def _full_seqr():
            s = await run_seqr(q, args, materialize=False)
            row.update(action="seqr", f1=s["f1"], precision=s["precision"],
                       recall=s["recall"],
                       time_s=row["time_s"] + s["time_s"],
                       cost_usd=row["cost_usd"] + s["cost_usd"])

        async def _contained_seqr():
            """Scoped refresh: prev-rule predicates appended to q_new's
            Sigma, engine dataset swapped to the first's materialized
            view (sigma_seqr)."""
            logger.info("Query %s contained in %s; SEQR with prev-"
                        "rule Sigma.", q, first)
            sem = deepcopy(ALL_QUERIES[q])
            sem.Sigma += [Predicate(c, op, t)
                          for c, t, op in prev_rules[0]]
            row["added_sigma"] = prev_rules[0]
            s = await run_seqr(
                q, args, sem=sem, materialize=False, reused_view=view)
            row.update(action="sigma_seqr", f1=s["f1"],
                       precision=s["precision"], recall=s["recall"],
                       time_s=row["time_s"] + s["time_s"],
                       cost_usd=row["cost_usd"] + s["cost_usd"])

        def _gate():
            start = time.time()
            relation, cost = aligned_with_first(client, q, first)
            row["aligned"] = relation
            row["time_s"] += time.time() - start
            row["cost_usd"] += cost

        def _try_regen() -> bool:
            """Selective router: cheap rule regeneration over the first's
            materialized columns. Verdict-independent by design (user
            ruling 08-23) — considered on ANY non-`same` verdict, never
            only when the gate cannot align. True = rewrite_sql applied."""
            start = time.time()
            supported, gate_cost = schema_supports(client, q, columns)
            row["schema_support"] = supported
            rules = None
            if supported:
                logger.info("Schema supports %s; regenerating UCQ.", q)
                rules, regen_cost = regen_rules(
                    client, q, first, prev_rules, columns)
                row["cost_usd"] += regen_cost
                logger.info("Regenerated UCQ for %s: %s", q, rules)
                if any(c[0] not in columns or c[2] not in ("<=", ">")
                       for conj in rules for c in conj):
                    logger.warning("Regenerated UCQ for %s is invalid "
                                   "(unknown column or op); falling "
                                   "back to SEQR.", q)
                    rules = None
            row["cost_usd"] += gate_cost
            row["time_s"] += time.time() - start
            if rules is not None:
                _apply("rewrite_sql", rules, journal=True)
                return True
            return False

        # ---- four mode routes ----------------------------------------
        if args.mode == "rerun":
            # Full-SEQR: no drift machinery; recompile q_new from scratch
            await _full_seqr()
        elif args.mode == "reuse":
            # Always-reuse: no gate; apply the first's rules blindly
            _apply("reuse", prev_rules)
        elif args.mode == "align":
            # Align only: 3-way gate; scoped SEQR on `contained`
            _gate()
            if row["aligned"] == "same":
                logger.info("Query %s same as %s; reusing rules.", q, first)
                _apply("reuse", prev_rules)
            elif row["aligned"] == "contained":
                await _contained_seqr()
            else:
                await _full_seqr()
        elif args.mode == "selective": 
            _gate()
            if row["aligned"] == "same":
                logger.info("Query %s same as %s; reusing rules.", q, first)
                _apply("reuse", prev_rules)
            elif _try_regen():
                pass  # rewrite_sql applied — served without any SEQR
            elif row["aligned"] == "contained":
                await _contained_seqr()
            else:
                logger.info("Falling back to SEQR for %s.", q)
                await _full_seqr()
        else:
            raise SystemExit(f"Unknown mode: {args.mode}")
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
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exp-group", default="_query_drift")
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
                f.write(json.dumps(row, default=float) + "\n")
        for row in rows:
            logger.info("[%s/%s] %s->%s %s: F1=%.4f $%.4f %.1fs",
                        group, args.mode, pair[0], row["q"], row["action"],
                        row["f1"] or 0.0, row["cost_usd"], row["time_s"])
    logger.info("Journal written to %s", out)


if __name__ == "__main__":
    asyncio.run(drift_main(build_parser().parse_args()))
