# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=unspecified-encoding,too-many-locals,too-many-branches
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=invalid-name
"""Preprocessing helpers for query reuse and Sigma refinement."""

import json
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pandas.api.types as ptypes

from data_structure import LdbDataManager, Predicate, SemCQ
from data_structure.llm_resp_templates import (
    PredicateResponse,
    PredicateResponses,
    RelevantFieldsResponse,
)
from llm import PROMPTS, LdbLLMClient

logger = logging.getLogger(__name__)


class Preprocessing:
    """Prepare workload data through query routing and Sigma refinement."""

    def __init__(
        self,
        llm_client: LdbLLMClient,
        data_manager: LdbDataManager,
        queries: dict[str, SemCQ],
        ckpt_path: Path,
        usage_statistics: list[dict[str, Any]],
        enable_cache: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.data_manager = data_manager
        self.queries = queries
        self.CKPT_path = ckpt_path
        self.usage_statistics = usage_statistics
        self.enable_cache = enable_cache

    def update_statistics(self, key: str, value: dict[str, Any]) -> None:
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for stat_key, stat_value in value.items():
            self.usage_statistics[0][key][stat_key] += stat_value

    def query_router(self, q1: SemCQ, q2: SemCQ) -> tuple[bool, dict]:
        from data_structure.llm_resp_templates import BooleanFeatureResponse

        q1_info = [
            f"Field: {p.field}, Success condition: {p.succ_cond}" for p in q1.Ps
        ]
        q2_info = [
            f"Field: {p.field}, Success condition: {p.succ_cond}" for p in q2.Ps
        ]

        q1_desc = "\n".join(q1_info) if q1_info else "No semantic predicates"
        q2_desc = "\n".join(q2_info) if q2_info else "No semantic predicates"

        prompt = f"""You are an expert in query optimization and result reuse.

Your task is to determine whether Query 1 can reuse the query results of
Query 2.

Query 1 can reuse Query 2's results if Query 2's results already contain or 
subsume the information needed by Query 1.
For example, if Query 2 retrieves all "romantic comedy" movies (which includes 
both romance and comedy), then Query 1 asking for all "comedy" movies can reuse 
Query 2's results.

=== Query 1 (wants to reuse results) ===
{q1_desc}

=== Query 2 (providing results) ===
{q2_desc}

=== Instructions ===

Analyze whether Query 1's requirements can be satisfied by Query 2's results. 
Consider:
1. Does Query 2's success condition cover or subsume Query 1's requirements?
2. If Query 2 returns items that satisfy a broader or related category, 
   do they include items that satisfy Query 1?
3. Are the fields compatible between the two queries?

Return True if Query 1 can reuse Query 2's results, False otherwise.
"""

        response = cast(
            BooleanFeatureResponse,
            self.llm_client.invoke(
                is_remote=True,
                modality="Text",
                prompt=prompt,
                response_model=BooleanFeatureResponse,
            ),
        )
        usage_stat = self.llm_client.get_usage_statistics()
        self.llm_client.reset_usage_statistics()

        return response.value, usage_stat

    def refine_sigma_satisfied_data(self) -> None:
        for q_name, sem_cq in self.queries.items():
            logger.info("Augmenting Sigma for query %s...", q_name)

            df_clean = self.data_manager.sigma_satisfied_data[0][q_name][
                "ldb_data"
            ].exclude_fk_and_id()
            schema_cols = set(df_clean.columns)
            schema_info = self._collect_table_schema(df_clean)
            query_desc = self._build_query_description(sem_cq)

            cache_path = self.CKPT_path / q_name / "prefilter_ucq.json"
            cached_results = (
                self._load_prefilter_cache(cache_path, schema_cols, q_name)
                if self.enable_cache
                else None
            )

            if cached_results is None:
                cached_results = self._generate_prefilter_cache(
                    q_name=q_name,
                    df_clean=df_clean,
                    query_desc=query_desc,
                    schema_info=schema_info,
                    cache_path=cache_path,
                )

            relevant_fields = {
                field
                for fields in cached_results.get("field_resp", {}).values()
                for field in fields
            }
            self.data_manager.relevant_base_features[q_name] = [
                field
                for field in self.data_manager.complete_dataset.base_features
                if field in relevant_fields
            ]

            new_ucq = self._expand_ucq(
                [
                    [PredicateResponse(**pred) for pred in group]
                    for group in cached_results["ucq_resp"]["value"]
                ]
            )
            new_ucq = [
                group
                for group in new_ucq
                if all(pred.field in schema_cols for pred in group)
            ]
            if not new_ucq:
                logger.info("No UCQ predicates suggested for query %s", q_name)

            self.data_manager.refine_sigma_satisfied_data(
                q_name=q_name, ucq=new_ucq
            )

    def _load_prefilter_cache(
        self, cache_path, schema_cols: set[str], q_name: str
    ) -> dict | None:
        if not cache_path.exists():
            return None

        with open(cache_path) as f:
            cached_results = json.load(f)

        cached_ucq = cached_results.get("ucq_resp", {}).get("value", [])
        cached_fields = set()
        for group in cached_ucq:
            for pred in group:
                if not isinstance(pred, dict):
                    continue
                field = pred.get("field")
                if isinstance(field, list):
                    cached_fields.update(field)
                elif field is not None:
                    cached_fields.add(field)

        if cached_fields.issubset(schema_cols):
            return cached_results

        logger.warning(
            "Cached UCQ for query %s is incompatible with current schema. "
            "Recomputing cache.",
            q_name,
        )
        cache_path.unlink(missing_ok=True)
        return None

    def _generate_prefilter_cache(
        self,
        q_name: str,
        df_clean: pd.DataFrame,
        query_desc: str,
        schema_info: str,
        cache_path,
    ) -> dict:
        logger.info(
            "Step 1: Identifying query-relevant fields for query %s...",
            q_name,
        )
        identify_fields_prompt = PROMPTS[
            "IDENTIFY_RELEVANT_FIELDS_PROMPT"
        ].format(
            query_desc=query_desc,
            schema_info=schema_info,
        )

        fields_response = cast(
            RelevantFieldsResponse,
            self.llm_client.invoke(
                is_remote=True,
                modality="Text",
                prompt=identify_fields_prompt,
                response_model=RelevantFieldsResponse,
            ),
        )
        self.update_statistics(
            "sigma_augmentation", self.llm_client.get_usage_statistics()
        )
        self.llm_client.reset_usage_statistics()

        relevant_fields_by_category = fields_response.value
        for category, fields in relevant_fields_by_category.items():
            relevant_fields_by_category[category] = [
                f for f in fields if f in df_clean.columns
            ]

        total_fields = sum(
            len(fields) for fields in relevant_fields_by_category.values()
        )
        logger.info(
            "Identified %s relevant fields across %s category/categories:",
            total_fields,
            len(relevant_fields_by_category),
        )
        for category, fields in relevant_fields_by_category.items():
            if fields:
                logger.info("  - %s: %s", category, fields)

        logger.info("Step 2: Generating UCQ for query %s...", q_name)
        generate_ucq_prompt = PROMPTS["GENERATE_UCQ_PROMPT"].format(
            query_desc=query_desc,
            relevant_fields=self._format_relevant_fields(
                relevant_fields_by_category
            ),
            schema_info=schema_info,
        )

        ucq_response = cast(
            PredicateResponses,
            self.llm_client.invoke(
                is_remote=True,
                modality="Text",
                prompt=generate_ucq_prompt,
                response_model=PredicateResponses,
            ),
        )
        self.update_statistics(
            "sigma_augmentation", self.llm_client.get_usage_statistics()
        )
        self.llm_client.reset_usage_statistics()

        cached_results = {
            "field_resp": relevant_fields_by_category,
            "ucq_resp": ucq_response.model_dump(),
        }
        if self.enable_cache:
            with open(cache_path, "w") as f:
                json.dump(cached_results, f, indent=2)
        return cached_results

    def _collect_table_schema(self, df: pd.DataFrame) -> str:
        """Collect and format table schema information for LLM prompt."""
        schema_lines = []

        for col in df.columns:
            if ptypes.is_numeric_dtype(df[col]):
                col_min = df[col].min()
                col_max = df[col].max()
                schema_lines.append(
                    f"- {col} (numerical): range [{col_min}, {col_max}]"
                )
                continue

            value_counts = df[col].value_counts()
            top_n = min(20, len(value_counts))
            top_values = value_counts.head(top_n).index.tolist()

            if len(value_counts) > 20:
                values_str = (
                    ", ".join(str(v) for v in top_values)
                    + f", ... ({len(value_counts)} total unique values)"
                )
            else:
                values_str = ", ".join(str(v) for v in top_values)

            schema_lines.append(f"- {col} (categorical): [{values_str}]")

        return "\n".join(schema_lines)

    def _build_query_description(self, sem_cq: SemCQ) -> str:
        """Build a query description from semantic predicates."""
        if not sem_cq.Ps:
            return "No semantic predicates provided."

        descriptions = []
        for sem_pred in sem_cq.Ps:
            descriptions.append(
                f"Field: {sem_pred.field}\n"
                f"Success condition: {sem_pred.succ_cond}\n"
                f"Prompt: {sem_pred.prompt}"
            )

        return "\n\n".join(descriptions)

    def _expand_ucq(
        self, ucq_responses: list[list[PredicateResponse]]
    ) -> list[list[Predicate]]:
        from itertools import product

        expanded_ucq = []

        for group in ucq_responses:
            if not group:
                continue

            eq_multi_preds = []
            other_preds = []

            for pred_resp in group:
                field_list = pred_resp.field
                value_list = pred_resp.value

                if pred_resp.op == "==" and (
                    len(field_list) > 1 or len(value_list) > 1
                ):
                    alternatives = []
                    for field in field_list:
                        for value in value_list:
                            alternatives.append(
                                Predicate(
                                    field=field, op=pred_resp.op, value=value
                                )
                            )
                    eq_multi_preds.append(alternatives)
                elif pred_resp.op == "!=" and (
                    len(field_list) > 1 or len(value_list) > 1
                ):
                    for field in field_list:
                        for value in value_list:
                            other_preds.append(
                                Predicate(
                                    field=field, op=pred_resp.op, value=value
                                )
                            )
                else:
                    other_preds.append(
                        Predicate(
                            field=field_list[0],
                            op=pred_resp.op,
                            value=value_list[0],
                        )
                    )

            if eq_multi_preds:
                for combo in product(*eq_multi_preds):
                    expanded_ucq.append(list(combo) + other_preds)
            else:
                expanded_ucq.append(other_preds)

        return expanded_ucq

    def _format_relevant_fields(
        self, fields_by_semantic_group: dict[str, list[str]]
    ) -> str:
        if not fields_by_semantic_group:
            return "No relevant fields identified"

        lines = []
        for group_name, fields in sorted(fields_by_semantic_group.items()):
            if fields:
                fields_str = ", ".join(fields)
                lines.append(f"- **{group_name}**: {fields_str}")

        return "\n".join(lines) if lines else "No relevant fields identified"
