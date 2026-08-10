# pylint: disable=missing-class-docstring,logging-fstring-interpolation
# pylint: disable=duplicate-code,missing-function-docstring
# pylint: disable=unspecified-encoding,unused-argument,invalid-name
# pylint: disable=too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Feature-space and coreset pipeline helpers."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, cast

from data_structure import (
    FeatureRefinementResponse,
    LdbDataManager,
    PopulationSpec,
    SemCQ,
)
from llm import PROMPTS, LdbLLMClient
from workloads.utils import (
    class_balanced_sample_weights,
    compute_feature_importance,
    encode_features,
)

from .feature_selection import select_feature_groups
from .semantic_features import ensure_semantic_features, required_feature_keys

logger = logging.getLogger(__name__)

STABLE_FEATURE_GENERATION_SCHEMA = 11


class FeaturePipeline:
    """Methods for feature construction, materialization, and trimming."""

    def __init__(
        self,
        data_manager: LdbDataManager,
        queries: dict[str, SemCQ],
        ckpt_path: Path,
        llm_client: LdbLLMClient,
        usage_statistics: list[dict[str, Any]],
        random_seed: int,
        b_lab: int,
        b_se: int,
        b_fs: int,
        enable_hitl: bool,
        enable_cache: bool = True,
    ) -> None:
        self.data_manager = data_manager
        self.queries = queries
        self.CKPT_path = ckpt_path
        self.llm_client = llm_client
        self.usage_statistics = usage_statistics
        self.random_seed = random_seed
        self.b_lab = b_lab
        self.b_se = b_se
        self.b_fs = b_fs
        self.enable_hitl = enable_hitl
        self.enable_cache = enable_cache

    def update_statistics(self, key: str, value: dict[str, Any]) -> None:
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for stat_key, stat_value in value.items():
            self.usage_statistics[0][key][stat_key] += stat_value

    def _stable_feature_space_context_key(
        self, q_name: str, feature_budget: int
    ) -> str:
        """Fingerprint a query-only feature proposal."""
        digest = hashlib.sha1()
        digest.update(
            json.dumps(
                {
                    "schema": STABLE_FEATURE_GENERATION_SCHEMA,
                    "feature_budget": feature_budget,
                    "query_predicates": [
                        {
                            "field": predicate.field,
                            "modality": predicate.modality,
                            "prompt": predicate.prompt,
                        }
                        for predicate in self.queries[q_name].Ps
                    ],
                    "remote_models": self.llm_client.config.get(
                        "REMOTE_MODELS", {}
                    ),
                    "inference": {
                        key: self.llm_client.config.get(key)
                        for key in (
                            "max_tokens",
                            "top_p",
                            "temperature",
                            "random_seed",
                        )
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _stable_feature_space_cache_path(
        self, q_name: str, feature_budget: int
    ) -> Path:
        """Return the shared query-only proposal cache path."""
        context_key = self._stable_feature_space_context_key(
            q_name, feature_budget
        )
        return (
            self.data_manager.annotation_ckpt_path
            / "feature_spaces"
            / q_name
            / f"{context_key}.json"
        )

    async def construct_feature_space(self) -> None:
        for q_name, sem_cq in self.queries.items():
            ckpt_path = self.CKPT_path / q_name / "feature_space.json"
            ckpt_usage_path = (
                self.CKPT_path / q_name / "usage_feature_space.json"
            )
            ckpt_context_path = (
                self.CKPT_path / q_name / "feature_space_context.json"
            )
            context_key = self._stable_feature_space_context_key(
                q_name, self.b_fs
            )
            cached_context_key = None
            if self.enable_cache and ckpt_context_path.exists():
                with open(ckpt_context_path) as context_file:
                    cached_context_key = json.load(context_file).get("key")

            if (
                self.enable_cache
                and ckpt_path.exists()
                and ckpt_usage_path.exists()
                and cached_context_key == context_key
            ):
                with open(ckpt_path) as f:
                    feature_space = json.load(f)
                with open(ckpt_usage_path) as f:
                    usage_statistics = json.load(f)
                logger.info(
                    "Loaded feature space from checkpoint for query %s.",
                    q_name,
                )
                cached_features = [
                    PopulationSpec(**spec) for spec in feature_space
                ]
                self.data_manager.enriched_features[q_name] = (
                    ensure_semantic_features(
                        q_name=q_name,
                        query=sem_cq,
                        feature_space=cached_features,
                    )
                )
                self.update_statistics("feature_space_init", usage_statistics)
                continue

            if self.enable_cache and (
                ckpt_path.exists() or ckpt_usage_path.exists()
            ):
                logger.warning(
                    "Ignoring feature-space checkpoint for query %s because "
                    "its annotation context does not match.",
                    q_name,
                )

            (
                feature_space,
                usage_statistics,
            ) = await self._initialize_feature_space(
                q_name=q_name,
                sem_cq=sem_cq,
                feature_budget=self.b_fs,
            )
            self.data_manager.enriched_features[q_name] = feature_space
            self.update_statistics("feature_space_init", usage_statistics)
            self.llm_client.reset_usage_statistics()
            if self.enable_cache:
                with open(ckpt_usage_path, "w") as f:
                    json.dump(usage_statistics, f, indent=2)
                with open(ckpt_context_path, "w") as context_file:
                    json.dump({"key": context_key}, context_file, indent=2)

        if self.enable_cache:
            for q_name in self.queries:
                ckpt_path = self.CKPT_path / q_name / "feature_space.json"
                feature_space = self.data_manager.enriched_features[q_name]
                with open(ckpt_path, "w") as f:
                    json.dump(
                        [spec.model_dump() for spec in feature_space],
                        f,
                        indent=2,
                    )
                logger.info(
                    "Saved feature space and usage statistics to checkpoint "
                    "for query %s.",
                    q_name,
                )

    async def sync_coreset_features(self, tag: str = "") -> None:
        """Materialize candidate features only on released annotations."""
        for q_name in self.queries:
            stat = await self.data_manager.sync_coreset_features(
                q_name, tag=tag, enable_cache=self.enable_cache
            )
            self.update_statistics("materialize_labeled_full", stat)
            self.llm_client.reset_usage_statistics()

    async def rank_and_trim_feature_space(self) -> None:
        ckpt_path = self.CKPT_path / "ranked_feature_space.json"

        importance_by_query: dict[str, dict[str, float]] = {}
        for q_name, coreset in self.data_manager.coresets.items():
            external_features = list(
                dict.fromkeys(
                    spec.target_col
                    for spec in self.data_manager.enriched_features[q_name]
                )
            )
            observed_size = coreset["observed_size"]
            X = encode_features(
                coreset["ldb_data"].df.loc[:, external_features]
            ).iloc[:observed_size]
            Y = coreset["labels"].iloc[:observed_size].astype(int)
            importance_weights = class_balanced_sample_weights(
                labels=Y,
                base_weight=coreset["annotation_weights"].iloc[:observed_size],
            )
            importance_by_query[q_name] = dict(
                compute_feature_importance(
                    X, Y, sample_weight=importance_weights
                ).itertuples(index=False)
            )

        selection = select_feature_groups(
            features_by_query=self.data_manager.enriched_features,
            importance_by_query=importance_by_query,
            required_exact_keys=required_feature_keys(self.queries),
            budget=self.b_se,
        )
        logger.info(
            "Selected %s logical external features "
            "(%s materialized aliases): %s",
            len(selection.selected_keys),
            len(selection.selected_feature_names),
            selection.selected_feature_names,
        )
        if self.enable_cache:
            with open(ckpt_path, "w") as f:
                json.dump(selection.ranked_feature_names, f, indent=2)

        self.data_manager.trimmed_feature_names = (
            selection.selected_feature_names
        )
        self.data_manager.enriched_features = selection.features_by_query

        for q_name in self.data_manager.coresets:
            await self.data_manager.sync_coreset_features(
                q_name, tag="trimmed", enable_cache=self.enable_cache
            )

        for q_name in self.data_manager.sigma_satisfied_data[0]:
            await self.data_manager.sync_sigma_satisfied_data_features(
                q_name,
                stream_idx=0,
                tag="trimmed",
                enable_cache=self.enable_cache,
            )

        materialized_external_features = set()
        for q_name in self.data_manager.coresets:
            ldb_data = self.data_manager.coresets[q_name]["ldb_data"]
            features = ldb_data.df.columns.tolist()
            original_schema = (
                ldb_data.base_features
                + ldb_data.id_features
                + ldb_data.foreign_keys
            )
            external_features = {
                feat for feat in features if feat not in original_schema
            }
            materialized_external_features.update(external_features)
            self.data_manager.enriched_features[q_name] = [
                spec
                for spec in self.data_manager.enriched_features[q_name]
                if spec.target_col in external_features
            ]

        self.data_manager.trimmed_feature_names = [
            feat
            for feat in self.data_manager.trimmed_feature_names
            if feat in materialized_external_features
        ]

    async def _initialize_feature_space(
        self,
        q_name: str,
        sem_cq: SemCQ,
        feature_budget: int,
    ) -> tuple[list[PopulationSpec], dict]:
        usage_statistics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
        }

        return self._initialize_stable_feature_space(
            q_name=q_name,
            sem_cq=sem_cq,
            feature_budget=feature_budget,
            usage_statistics=usage_statistics,
        )

    def _initialize_stable_feature_space(
        self,
        q_name: str,
        sem_cq: SemCQ,
        feature_budget: int,
        usage_statistics: dict[str, float],
    ) -> tuple[list[PopulationSpec], dict[str, float]]:
        """Build or reuse a seed-independent semantic feature proposal."""
        cache_path = self._stable_feature_space_cache_path(
            q_name, feature_budget
        )
        if self.enable_cache and cache_path.exists():
            with open(cache_path) as cache_file:
                cached = json.load(cache_file)
            feature_space = [PopulationSpec(**spec) for spec in cached]
            logger.info(
                "Loaded stable query-only feature space for query %s.",
                q_name,
            )
            return (
                ensure_semantic_features(q_name, sem_cq, feature_space),
                usage_statistics,
            )

        feature_space = []
        existing_targets = set()
        predicate_budgets = self._predicate_feature_budgets(
            predicate_count=len(sem_cq.Ps),
            total_budget=feature_budget,
        )
        for sem_pred, predicate_budget in zip(
            sem_cq.Ps, predicate_budgets, strict=True
        ):
            derived_budget = predicate_budget - 1
            if derived_budget <= 0:
                continue
            prompt = self._build_feature_generation_prompt(
                sem_pred=sem_pred, feature_budget=derived_budget
            )
            response = cast(
                FeatureRefinementResponse,
                self.llm_client.invoke(
                    modality=sem_pred.modality,
                    is_remote=True,
                    prompt=prompt,
                    response_model=FeatureRefinementResponse,
                ),
            )
            review_prompt = self._build_feature_review_prompt(
                sem_pred=sem_pred,
                feature_budget=derived_budget,
                candidates=response.to_add,
            )
            response = cast(
                FeatureRefinementResponse,
                self.llm_client.invoke(
                    modality=sem_pred.modality,
                    is_remote=True,
                    prompt=review_prompt,
                    response_model=FeatureRefinementResponse,
                ),
            )
            redundancy_prompt = self._build_feature_redundancy_prompt(
                candidates=response.to_add
            )
            redundancy_review = cast(
                FeatureRefinementResponse,
                self.llm_client.invoke(
                    modality=sem_pred.modality,
                    is_remote=True,
                    prompt=redundancy_prompt,
                    response_model=FeatureRefinementResponse,
                    model_id=0,
                ),
            )
            redundant_targets = set(redundancy_review.to_remove)
            accepted = 0
            for spec in response.to_add:
                if accepted >= derived_budget:
                    break
                if not self._valid_derived_spec(spec, sem_pred):
                    continue
                if spec.target_col in redundant_targets:
                    continue
                if spec.target_col in existing_targets:
                    continue
                feature_space.append(spec)
                existing_targets.add(spec.target_col)
                accepted += 1
            for key, value in self.llm_client.get_usage_statistics().items():
                usage_statistics[key] += value
            self.llm_client.reset_usage_statistics()

        feature_space = ensure_semantic_features(
            q_name=q_name, query=sem_cq, feature_space=feature_space
        )
        if self.enable_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as cache_file:
                json.dump(
                    [spec.model_dump() for spec in feature_space],
                    cache_file,
                    indent=2,
                )
            logger.info(
                "Saved stable query-only feature space for query %s.", q_name
            )
        return feature_space, usage_statistics

    @staticmethod
    def _predicate_feature_budgets(
        predicate_count: int, total_budget: int
    ) -> list[int]:
        """Allocate one shared feature budget across semantic predicates."""
        if predicate_count <= 0:
            return []
        effective_budget = max(total_budget, predicate_count)
        base, remainder = divmod(effective_budget, predicate_count)
        return [
            base + (position < remainder) for position in range(predicate_count)
        ]

    @staticmethod
    def _build_feature_review_prompt(
        sem_pred, feature_budget: int, candidates: list[PopulationSpec]
    ) -> str:
        """Request a complete atomic replacement for one proposal."""
        candidate_payload = json.dumps(
            [spec.model_dump() for spec in candidates], indent=2
        )
        return (
            "Audit this query-only derived feature proposal. Reconstruct it "
            "as a complete, complementary atomic feature set.\n\n"
            f"Semantic task: {sem_pred.prompt}\n"
            f"Source field: {sem_pred.field}\n"
            f"Source modality: {sem_pred.modality}\n"
            f"Feature budget: {feature_budget}\n\n"
            f"Candidate specifications:\n{candidate_payload}\n\n"
            "Each feature must test exactly one observable cue or one set of "
            "true synonyms. Preserve every distinct observation named "
            "anywhere in the candidates, but split any feature that combines "
            "observations which can occur independently. An umbrella "
            "category, an OR-joined target name, or a prompt enumerating "
            "distinct manifestations is invalid. Use the available budget "
            "first for distinct positive ways the task can hold, then for "
            "context or exclusion evidence. Use only the semantic task and "
            "candidates; no data examples, annotations, prevalence, "
            "evaluation labels, or benchmark knowledge are available. "
            f"Return exactly {feature_budget} complete specifications in "
            "to_add and an empty to_remove list. Every source_col and "
            "source_modality must exactly match the values above. Output "
            "valid JSON only."
        )

    @staticmethod
    def _build_feature_redundancy_prompt(
        candidates: list[PopulationSpec],
    ) -> str:
        """Identify only later candidates equivalent to an earlier one."""
        candidate_payload = json.dumps(
            [spec.model_dump() for spec in candidates], indent=2
        )
        return (
            "Audit the candidate specifications below only for semantic "
            "redundancy. A later candidate is redundant when it asks the "
            "same observable question as an earlier candidate using "
            "synonyms, paraphrases, alternate terminology, or a "
            "restatement, so both should return the same value for "
            "essentially every possible source value. Related cues, "
            "correlated cues, umbrella/subtype relationships, and "
            "observations that can occur independently are not redundant. "
            "Preserve the earliest candidate in each equivalent group. "
            "Return an empty to_add list. In to_remove, return the "
            "target_col of every later redundant candidate, copied exactly. "
            "Do not rewrite, replace, merge, split, or add any feature. If "
            "there are no redundant pairs, return both lists empty. Use "
            "only these specifications; no examples, labels, prevalence, "
            "or benchmark information are available. Output valid JSON "
            "only.\n\n"
            f"Candidate specifications:\n{candidate_payload}"
        )

    @staticmethod
    def _valid_derived_spec(spec: PopulationSpec, sem_pred) -> bool:
        """Accept only executable features tied to the requested source."""
        return (
            spec.source_col == sem_pred.field
            and spec.source_modality == sem_pred.modality
            and spec.feature_type in {"bool", "float", "int"}
            and bool(spec.target_col.strip())
            and bool(spec.prompt.strip())
            and " ".join(spec.prompt.split()).casefold()
            != " ".join(sem_pred.prompt.split()).casefold()
        )

    @staticmethod
    def _build_feature_generation_prompt(sem_pred, feature_budget: int) -> str:
        """Build a query-only prompt for atomic evidence features."""
        return PROMPTS["GEN_FEAT_CANDIDATE_PROMPT"].format(
            MODALITY=sem_pred.modality,
            DESC=sem_pred.prompt,
            SOURCE_COL=sem_pred.field,
            FEATURE_BUDGET=feature_budget,
            SCHEMA_SAMPLE_SECTION="",
            PREVIOUS_FEATURES_SECTION="",
            PERFORMANCE_FEEDBACK_SECTION="",
            INSTRUCTIONS_SECTION=(
                "Decompose the semantic task into complementary, nonredundant "
                "observable evidence dimensions that are directly implied by "
                "the task and extractable from one source value. Each feature "
                "must test one atomic cue: it may group true synonyms, but "
                "must not merge distinct manifestations, relations, causes, "
                "severity levels, or alternative ways the task can hold. "
                "Cover distinct ways the requested condition can appear, "
                "including less frequent but directly relevant cues. When "
                "the task supports enough distinct evidence, propose exactly "
                f"{feature_budget} features. Prefer direct, precise evidence "
                "over broad correlates. Do not assume access to dataset "
                "examples, individual annotations, evaluation labels, or "
                "benchmark-specific knowledge. Do not invent demographic or "
                "contextual proxies that are not part of the task."
            ),
            CONSTRAINTS_ADDITIONAL="",
        )
