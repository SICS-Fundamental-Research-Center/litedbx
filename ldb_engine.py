import logging
from typing import Tuple
from workloads.ldb_workload import LdbWorkload
from time import time

logger = logging.getLogger(__name__)


class LdbEngine:
    def __init__(self, workload: LdbWorkload) -> None:
        self.workload = workload
        self.dynamic_setting = workload.dynamic_setting



    async def p_eval(self, debug=False):
        logger.info((
            "Launching LiteDBX Engine for queries: "
            f"{list(self.workload.queries.keys())} "
            f"in scenario: {self.workload.scenario} with "
            f"dynamic setting: {self.dynamic_setting}."
        ))
        logger.info(f"Start PEval with {self.dynamic_setting[0] * 100}%-data partition.")

        """
        [Phase 1] Preprocessing.
        """
        preprocessing_start = time()
        # (1.1) Initialize the data stream for the dynamic setting.
        self.workload.data_manager.init_data_stream()

        # (1.2) Sigma retrieval.
        self.workload.data_manager.init_sigma_satisfied_data()

        # (1.3) [Optional] Augment the Sigma with high-confidence 
        #   LLM-suggested predicates to narrow down the data scale.
        self.workload.refine_sigma_satisfied_data()
        preprocessing_end = time()
        preprocessing_duration = preprocessing_end - preprocessing_start

        """
        [Phase 2] Construct the feature space and the coreset.
        """
        coreset_start = time()
        # (2.1) Construct the feature space.
        await self.workload.construct_feature_space(debug=debug)
        coreset_fs_end = time()
        coreset_fs_duration = coreset_fs_end - coreset_start

        # (2.2) Populate the dataset with the selected features.
        await self.workload.sync_with_enriched_features(tag="init")
        coreset_fs_sync_end = time()
        coreset_fs_sync_duration = coreset_fs_sync_end - coreset_fs_end

        # (2.3) Expand the coreset.
        self.workload.expand_coresets(inc_round=0)
        coreset_end = time()
        coreset_expand_duration = coreset_end - coreset_fs_sync_end
        coreset_duration = coreset_end - coreset_start

        """
        [Phase 3] Schema selection and query rewriting.
        """
        qr_start = time()
        # (3.1) Rank and trim the feature space according the feature selection budget.
        await self.workload.rank_and_trim_feature_space()
        qr_trim_end = time()
        qr_trim_duration = qr_trim_end - qr_start

        # (3.2) Select the best schema and translate the query.
        execution_trace = {}
        if not self.workload.enable_rewrite:
            await self.workload.rewrite_and_execute_query_noRew()
        elif not self.workload.enable_enrich:
            self.workload.rewrite_and_execute_query_noEnr()
        else:
            _, execution_trace = self.workload.rewrite_and_execute_query()
        qr_rewrite_end = time()
        qr_rewrite_duration = qr_rewrite_end - qr_trim_end
        qr_duration = qr_rewrite_end - qr_start

        logger.info(f"Preprocessing duration: {preprocessing_duration:.2f} seconds")
        logger.info(f"Feature space construction duration: {coreset_fs_duration:.2f} seconds")
        logger.info(f"Feature space sync duration: {coreset_fs_sync_duration:.2f} seconds")
        logger.info(f"Coreset expansion duration: {coreset_expand_duration:.2f} seconds")
        logger.info(f"Total coreset construction duration: {coreset_duration:.2f} seconds")
        logger.info(f"Query rewriting trim duration: {qr_trim_duration:.2f} seconds")
        logger.info(f"Query rewriting duration: {qr_rewrite_duration:.2f} seconds")
        logger.info(f"Total query rewriting duration: {qr_duration:.2f} seconds")

        return execution_trace



    async def inc_eval(self) -> Tuple[bool, list]:
        if len(self.dynamic_setting) <= 1:
            logger.info("No incremental evaluation setting found. Skipping inc_eval.")
            return False, []

        rerun, result = await self.workload.incremental_processing()

        if result == []:
            logger.info("No incremental processing results to report.")

        return rerun, result



    async def execute(self, debug=False):
        execution_trace = await self.p_eval(debug=debug)
        
        _, results = await self.inc_eval()

        """
        Report the execution results.
        """
        self.workload._report_evaluation_trace(execution_trace)
        
        if len(results) > 0:
            self.workload._report_dynamic_results(results)

        self.workload._report_usage_statistics()

        return {
            "scenario": self.workload.scenario,
            "queries": list(self.workload.queries.keys()),
            "dynamic_setting": self.dynamic_setting,
            "execution_trace": execution_trace,
            "incremental_results": results,
            "usage_statistics": self.workload.usage_statistics[0],
        }
