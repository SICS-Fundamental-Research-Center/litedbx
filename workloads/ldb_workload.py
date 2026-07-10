# pylint: disable=invalid-name,too-many-instance-attributes
# pylint: disable=missing-function-docstring
"""Engine-facing LiteDBX workload facade."""

import logging
from pathlib import Path
from time import time
from typing import Any

from data_structure import LdbDataManager, SemCQ
from llm import LdbLLMClient
from workloads.core.coreset_maintainer import CoresetMaintainer
from workloads.core.feature_pipeline import FeaturePipeline
from workloads.core.preprocessing import Preprocessing
from workloads.core.query_execution import QueryExecution
from workloads.core.reporting import Reporting

logger = logging.getLogger(__name__)


class LdbWorkload:
    """LiteDBX workload API consumed by :class:`ldb_engine.LdbEngine`."""

    def __init__(
        self,
        data_dir: str,
        scenario: str,
        queries: dict[str, SemCQ],
        config: dict,
    ) -> None:
        self.llm_client = LdbLLMClient()

        self.data_dir = data_dir
        self.scenario = scenario
        self.queries = queries
        self.config = config
        self._load_config_values()

        self.CKPT_path = (
            Path(__file__).parent.parent
            / ".data_ckpt"
            / self.scenario
            / "_".join(str(step) for step in self.dynamic_setting)
        )
        self._ensure_query_ckpts()

        self.usage_statistics = self._init_usage_statistics(
            self.dynamic_setting
        )

        self._init_components()

    def _init_components(self) -> None:
        self._preprocessing = Preprocessing(
            llm_client=self.llm_client,
            data_manager=self.data_manager,
            queries=self.queries,
            ckpt_path=self.CKPT_path,
            usage_statistics=self.usage_statistics,
        )
        self._feature_pipeline = FeaturePipeline(
            data_manager=self.data_manager,
            queries=self.queries,
            ckpt_path=self.CKPT_path,
            llm_client=self.llm_client,
            usage_statistics=self.usage_statistics,
            random_seed=self.random_seed,
            b_lab=self.b_lab,
            b_se=self.b_se,
        )
        self._coreset_maintainer = CoresetMaintainer(
            data_manager=self.data_manager,
            config=self.config,
            enable_conf_pred=self.enable_conf_pred,
            enable_conf_struct=self.enable_conf_struct,
        )
        self._query_execution = QueryExecution(
            coreset_maintainer=self._coreset_maintainer,
            data_manager=self.data_manager,
            queries=self.queries,
            config=self.config,
            llm_client=self.llm_client,
            dynamic_setting=self.dynamic_setting,
            b_rew=self.b_rew,
            b_lab=self.b_lab,
            delta=self.delta,
            eta=self.eta,
        )
        self._reporting = Reporting(self.usage_statistics)

    @staticmethod
    def _init_usage_statistics(dynamic_setting: list[float]) -> list[dict]:
        return [
            {
                item: {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "prompt_cost": 0.0,
                    "completion_cost": 0.0,
                    "total_cost": 0.0,
                }
                for item in [
                    "sigma_augmentation",
                    "feature_space_init",
                    "feature_space_refine",
                    "materialize_labeled_full",
                    "materialize_unlabeled_full",
                    "materialize_unlabeled_inc",
                ]
            }
            for _ in range(len(dynamic_setting))
        ]

    # ------------------------------------------------------------------
    # Public engine-facing API
    # ------------------------------------------------------------------

    async def run_partial_evaluation(
        self, debug: bool = False
    ) -> tuple[dict[str, Any], dict[str, float]]:
        durations: dict[str, float] = {}

        phase_start = time()
        self.data_manager.init_data_stream()
        self.data_manager.init_sigma_satisfied_data()
        self._preprocessing.refine_sigma_satisfied_data()
        durations["preprocessing"] = time() - phase_start

        phase_start = time()
        step_start = phase_start
        await self._feature_pipeline.construct_feature_space(debug=debug)
        step_end = time()
        durations["coreset_feature_space"] = step_end - step_start

        step_start = step_end
        await self._feature_pipeline.sync_with_enriched_features(tag="init")
        step_end = time()
        durations["coreset_sync"] = step_end - step_start

        step_start = step_end
        self._coreset_maintainer.expand_coresets(inc_round=0)
        step_end = time()
        durations["coreset_expand"] = step_end - step_start
        durations["coreset_total"] = step_end - phase_start

        phase_start = time()
        step_start = phase_start
        await self._feature_pipeline.rank_and_trim_feature_space()
        step_end = time()
        durations["query_trim"] = step_end - step_start

        step_start = step_end
        execution_trace = await self._query_execution.execute_queries(
            enable_rewrite=self.enable_rewrite,
            enable_enrich=self.enable_enrich,
            debug=debug,
        )
        step_end = time()
        durations["query_rewrite"] = step_end - step_start
        durations["query_total"] = step_end - phase_start

        return execution_trace, durations

    async def run_incremental_evaluation(self) -> tuple[bool, list[Any]]:
        if len(self.dynamic_setting) <= 1:
            logger.info(
                "No incremental evaluation setting found. Skipping inc_eval."
            )
            return False, []

        rerun, result = await self._query_execution.incremental_processing()
        if not result:
            logger.info("No incremental processing results to report.")
        return rerun, result

    def report_results(
        self, execution_trace: dict[str, Any], results: list[Any]
    ) -> None:
        self._reporting.report_evaluation_trace(execution_trace)
        if results:
            self._reporting.report_dynamic_results(results)
        self._reporting.report_usage_statistics()

    def build_result_payload(
        self,
        execution_trace: dict[str, Any],
        phase_durations: dict[str, float],
        incremental_results: list[Any],
    ) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "queries": list(self.queries.keys()),
            "dynamic_setting": self.dynamic_setting,
            "execution_trace": execution_trace,
            "phase_durations": phase_durations,
            "incremental_results": incremental_results,
            "usage_statistics": self.usage_statistics[0],
        }

    def inject_exp_setting(self, exp_group: str, exp_patch: dict) -> None:
        assert exp_group != "", (
            "Exp group cannot be empty when injecting exp setting."
        )
        assert exp_patch is not None, (
            "Exp patch cannot be None when injecting exp setting."
        )
        for k, v in exp_patch.items():
            assert k in self.config, f"Invalid config key in exp patch: {k}"
            self.config[k] = v
            logger.info("Exp patch applied: %s=%s", k, v)

        self._load_config_values()

        exp_term = "_".join(
            [
                str(v[0]) + "=" + str(v[1])
                for v in list(
                    zip(exp_patch.keys(), exp_patch.values(), strict=True)
                )
            ]
        )
        self.CKPT_path = (
            Path(__file__).parent.parent
            / ".data_ckpt"
            / exp_group
            / self.scenario
            / exp_term
            / "_".join(str(step) for step in self.dynamic_setting)
        )
        self._ensure_query_ckpts()
        self.data_manager.set_ckpt_path(self.CKPT_path)
        self._init_components()

    def _load_config_values(self) -> None:
        self.random_seed = self.config["random_seed"]
        self.b_lab = self.config["b_lab"]
        self.b_se = self.config["b_se"]
        self.b_rew = self.config["b_rew"]
        self.b_fs = self.config["b_fs"]
        self.eta = self.config["eta"]
        self.delta = self.config["delta"]
        self.enable_hitl = self.config["enable_hitl"]
        self.enable_conf_pred = self.config["enable_conf_pred"]
        self.enable_conf_struct = self.config["enable_conf_struct"]
        self.enable_enrich = self.config["enable_enrich"]
        self.enable_rewrite = self.config["enable_rewrite"]
        self.dynamic_setting = self.config["dynamic_setting"]
        self.data_manager = LdbDataManager(
            data_dir=self.data_dir,
            scenario=self.scenario,
            queries=self.queries,
            llm_client=self.llm_client,
            dynamic_steps=self.dynamic_setting,
        )

    def _ensure_query_ckpts(self) -> None:
        for q_name in self.queries:
            (self.CKPT_path / q_name).mkdir(parents=True, exist_ok=True)

    def update_statistics(self, key: str, value: dict) -> None:
        assert key in self.usage_statistics[0], f"Invalid statistics key: {key}"
        for stat_key, stat_value in value.items():
            self.usage_statistics[0][key][stat_key] += stat_value

    def _update_statistics(self, key: str, value: dict) -> None:
        self.update_statistics(key, value)
