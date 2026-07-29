# pylint: disable=duplicate-code
"""LiteDBX data lifecycle manager."""

from pathlib import Path

import pandas as pd

from llm import LdbLLMClient

from .coreset import CoresetStore
from .data_stream import DataStream
from .ldb_data import LdbData
from .llm_resp_templates import PopulationSpec
from .sem_query import Predicate, SemCQ
from .sigma_satisfied_data import SigmaSatisfiedData


class LdbDataManager:  # pylint: disable=too-many-instance-attributes
    """Manage LiteDBX data streams, Sigma-filtered data, and coresets."""

    STREAM_SEED = 42

    def __init__(
        self,
        data_dir: str,
        scenario: str,
        queries: dict[str, SemCQ],
        llm_client: LdbLLMClient,
        dynamic_steps: list[float],
    ):
        self.data_dir = data_dir
        self.scenario = scenario
        self.complete_dataset = LdbData(data_dir=data_dir)
        self.queries = queries
        self.llm_client = llm_client
        self.dynamic_steps = dynamic_steps

        self.data_stream = DataStream()
        self.sigma_satisfied_data = SigmaSatisfiedData()
        self.coresets = CoresetStore()

        self.enriched_features: dict[str, list[PopulationSpec]] = {}
        self.trimmed_feature_names: list[str] = []
        self.rewrite_rules: dict[str, dict] = {}

        self.ckpt_path = self._default_ckpt_path()
        self.annotation_ckpt_path = self._default_annotation_ckpt_path()
        self._ensure_ckpt_path()
        self.annotation_ckpt_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public workflow API
    # ------------------------------------------------------------------

    def init_data_stream(self) -> None:
        """Build data stream partitions from the complete dataset."""
        self.data_stream.init(
            complete_dataset=self.complete_dataset,
            dynamic_steps=self.dynamic_steps,
            seed=self.STREAM_SEED,
        )

    def init_sigma_satisfied_data(self) -> None:
        """Retrieve Sigma-satisfied data and initialize ground-truth labels."""
        self.sigma_satisfied_data.initialize(
            data_stream=self.data_stream,
            queries=self.queries,
            complete_config=self.complete_dataset.config,
            data_dir=self.data_dir,
        )

    def refine_sigma_satisfied_data(
        self, q_name: str, ucq: list[list[Predicate]]
    ) -> None:
        """Refine Sigma-satisfied data for one query across streams."""
        self.sigma_satisfied_data.refine(
            q_name=q_name,
            ucq=ucq,
            queries=self.queries,
            complete_config=self.complete_dataset.config,
            data_dir=self.data_dir,
        )

    async def acquire_annotation_and_init_coreset(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        b_lab: int,
        seed: int = 42,
        use_hitl: bool = True,
    ) -> None:
        """Acquire labels and initialize query coresets."""
        await self.coresets.acquire_annotation_and_init(
            queries=self.queries,
            sigma_satisfied_data=self.sigma_satisfied_data,
            complete_config=self.complete_dataset.config,
            llm_client=self.llm_client,
            ckpt_root=self.ckpt_path,
            pseudo_ckpt_root=self.annotation_ckpt_path,
            b_lab=b_lab,
            feature_spaces=self.enriched_features,
            seed=seed,
            use_hitl=use_hitl,
        )

    async def sync_coreset_features(
        self,
        q_name: str,
        tag: str = "",
        enable_cache: bool = True,
        is_remote: bool = False,
    ) -> dict:
        """Synchronize enriched features for one query coreset."""
        return await self.coresets.sync_features(
            q_name=q_name,
            enriched_features=self.enriched_features,
            llm_client=self.llm_client,
            ckpt_root=self.ckpt_path,
            tag=tag,
            enable_cache=enable_cache,
            is_remote=is_remote,
        )

    async def sync_sigma_satisfied_data_features(
        self,
        q_name: str,
        tag: str = "",
        stream_idx: int = 0,
        enable_cache: bool = True,
        is_remote: bool = False,
    ) -> dict:
        """Synchronize enriched features for one Sigma-satisfied dataset."""
        return await self.sigma_satisfied_data.sync_features(
            q_name=q_name,
            stream_idx=stream_idx,
            enriched_features=self.enriched_features,
            llm_client=self.llm_client,
            ckpt_root=self.ckpt_path,
            tag=tag,
            enable_cache=enable_cache,
            is_remote=is_remote,
        )

    def eval_query_quality(
        self,
        q_name: str,
        selected_cols: list[str],
        stream_idx: int,
        pred_labels: list[pd.Series],
    ) -> dict:
        """Evaluate predicted labels against query ground truth."""
        return self.sigma_satisfied_data.eval_query_quality(
            q_name=q_name,
            selected_cols=selected_cols,
            stream_idx=stream_idx,
            pred_labels=pred_labels,
        )

    # ------------------------------------------------------------------
    # Checkpoint API
    # ------------------------------------------------------------------

    def set_ckpt_path(self, path: Path) -> None:
        """Set the checkpoint root and ensure it exists."""
        self.ckpt_path = Path(path)
        self._ensure_ckpt_path()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _default_ckpt_path(self) -> Path:
        """Return the default checkpoint root for this data manager."""
        return (
            Path(__file__).parent.parent
            / ".data_ckpt"
            / self.scenario
            / "_".join(str(step) for step in self.dynamic_steps)
        )

    def _default_annotation_ckpt_path(self) -> Path:
        """Return a cache root shared across labeling-budget variants."""
        return (
            Path(__file__).parent.parent
            / ".data_ckpt"
            / "annotation_cache"
            / self.scenario
            / "_".join(str(step) for step in self.dynamic_steps)
        )

    def _ensure_ckpt_path(self) -> None:
        """Ensure the manager checkpoint root exists."""
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
