import logging
from typing import Tuple
from workloads.ldb_workload import LdbWorkload

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
        # (1.1) Initialize the data stream for the dynamic setting.
        self.workload.data_manager.init_data_stream()

        # (1.2) Sigma retrieval.
        self.workload.data_manager.init_sigma_satisfied_data()

        # (1.3) [Optional] Augment the Sigma with high-confidence 
        #   LLM-suggested predicates to narrow down the data scale.
        self.workload.refine_sigma_satisfied_data()



        """
        [Phase 2] Construct the feature space and the coreset.
        """
        # (2.1) Construct the feature space.
        await self.workload.construct_feature_space(debug=debug)

        # (2.2) Populate the dataset with the selected features.
        await self.workload.sync_with_enriched_features(tag="init")

        # (2.3) Expand the coreset.
        self.workload.expand_coresets(inc_round=0)

        """
        [Phase 3] Schema selection and query rewriting.
        """
        # (3.1) Rank and trim the feature space according the feature selection budget.
        await self.workload.rank_and_trim_feature_space()

        # (3.2) Select the best schema and translate the query.
        _, execution_trace = self.workload.rewrite_and_execute_query()

        return execution_trace



    async def inc_eval(self) -> Tuple[bool, list]:
        if len(self.dynamic_setting) <= 1:
            logger.info("No incremental evaluation setting found. Skipping inc_eval.")
            return False, []

        rerun, result = await self.workload.incremental_processing()

        if result == []:
            logger.info("No incremental processing results to report.")

        return rerun, result


        # TODO: Add processing for the case when rerun is True.





    async def execute(self, debug=False):
        execution_trace = await self.p_eval(debug=debug)
        
        rerun, results = await self.inc_eval()

        """
        Report the execution results.
        """
        self.workload._report_evaluation_trace(execution_trace)
        
        if len(results) > 0:
            self.workload._report_dynamic_results(results)

        self.workload._report_usage_statistics()


        # logger.info((
        #     f"Executing LDB Worker for queries: {list(self.workload.queries.keys())} "
        #     f"in scenario: {self.workload.scenario}"
        # ))

        # # Phase (1.1): Preprocessing
        # self.workload.apply_sigma(debug=debug)
        # for q_name, df in self.workload.sigma_satisfied_data[0].items():
        #     logger.info(f"Sigma retrieval for query '{q_name}' resulted in {len(df["data"].df)} rows.")

        # # Phase (1.2): Augment Sigma with LLM-suggested predicates
        # self.workload.augment_sigma_and_apply()

        # # Phase (2.1): Initialize the feature space and prepare the coreset.
        # # TODO: Add LLM-generated pseudo-label to the feature space.
        # # TODO: Evaluate the soundness of the LLM-generated pseudo-label.
        # await self.workload.init_coresets(debug=debug)

        # # Phase (2.2): Materialize the remaining part.
        # # TODO: Evaluate the soundness of logits of LLM-generated pseudo-labels.
        # # TODO: If the logits is reliable, try model cascading.
        # # TODO: Try batched prompting (row/col-wise) to accelerate the population.
        # # TODO: Try three-level labelling (negative, not sure, positive).
        # # TODO: Explore the impact of the selectivity.
        # await self.workload.populate_unlabeled_data()

        # # Phase (2.3): Expand the coreset.
        # self.workload.expand_coreset(debug=False)

        # # Phase (3.1): Generate the candidate external features for each dataset.
        # self.workload.generate_candidate_external_features()

        # # Phase (3.2): Select the best schema and return the rewritten query.
        # best_statistics, execution_trace = \
        #     self.workload.p_eval(debug=False)

        # self.workload._report_evaluation_trace(execution_trace)

        # self.workload._report_usage_statistics()

        # # Dynamic processing.
        # eval_results = await self.workload.inc_eval(debug=debug)
        # self.workload._report_dynamic_results(eval_results)

