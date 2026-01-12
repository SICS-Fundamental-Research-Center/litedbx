from time import time
import asyncio
import logging
import sys
from workloads import medical_workloads


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    workloads = ["Q1"]

    ldb_engine = medical_workloads.build_query_engine(
        workloads=workloads,
        feature_enrich_budget=3,
        query_rewrite_budget=3,
    )

    start = time()

    asyncio.run(
        ldb_engine.apply(
            queries=workloads,
            enable_proxies=False
        )
    )

    end = time()
    logger.info(f"Total execution time: {end - start} seconds")



