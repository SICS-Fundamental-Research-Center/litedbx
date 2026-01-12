from time import time
import asyncio
import logging
from workloads import medical_workloads
from logger_config import setup_logger


if __name__ == "__main__":

    # Set up logging
    logger = setup_logger(level=logging.INFO)

    workloads = ["Q1"]

    ldb_engine = medical_workloads.build_query_engine(
        workloads=workloads,
        feature_enrich_budget=3,
        query_rewrite_budget=3,
    )

    start = time()

    asyncio.run(
        ldb_engine.apply(workloads)
    )

    end = time()
    logger.info(f"Total execution time: {end - start} seconds")



