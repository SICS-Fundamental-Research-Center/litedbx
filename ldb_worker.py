import logging
from workloads.ldb_workload import LdbWorkload

logger = logging.getLogger(__name__)


class LdbWorker:
    def __init__(self, workload: LdbWorkload) -> None:
        self.workload = workload

    async def execute(self, debug=False):
        logger.info((
            f"Executing LDB Worker for queries: {list(self.workload.queries.keys())} "
            f"in scenario: {self.workload.scenario}"
        ))

        # Phase (1.1): Preprocessing
        self.workload.apply_sigma(debug=debug)
        for q_name, df in self.workload.sigma_satisfied_data.items():
            logger.info(f"Sigma retrieval for query '{q_name}' resulted in {len(df["data"].df)} rows.")

        # Phase (2.1): Initialize the feature space and prepare the coreset.
        # TODO: Add LLM-generated pseudo-label to the feature space.
        # TODO: Evaluate the soundness of the LLM-generated pseudo-label.
        await self.workload.init_coresets(enable_refinement=False, debug=debug)

        # Phase (2.2): Materialize the remaining part.
        # TODO: Evaluate the soundness of logits of LLM-generated pseudo-labels.
        # TODO: If the logits is reliable, try model cascading.
        # TODO: Try batched prompting (row/col-wise) to accelerate the population.
        # TODO: Try three-level labelling (negative, not sure, positive).
        # TODO: Explore the impact of the selectivity.
        await self.workload.populate_unlabeled_data()

        # Phase (2.3): Expand the coreset.
        self.workload.expand_coreset(debug=True)

        # Phase (3.1): Generate the candidate external features for each dataset.
        self.workload.generate_candidate_external_features()
        
