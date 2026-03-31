from time import time
import logging
import sys
from workloads import (
    medical,
    movie,
    ecomm,
    mmqa,
    animals,
)
from ldb_engine import LdbEngine
import asyncio
    
ENABLE_DEBUG=True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    workload = "mmqa"
    queries = [
        "Q3f", 
    ]

    workload_mapping = {
        "medical": medical.get_workload,
        "movie": movie.get_workload,
        "ecomm": ecomm.get_workload,
        "mmqa": mmqa.get_workload,
        "animals": animals.get_workload,
    }

    workload_func = workload_mapping.get(workload)
    assert workload_func is not None, f"Invalid workload: {workload}"
    workload = workload_func(queries=queries)

    ldb_engine = LdbEngine(workload)

    start = time()

    asyncio.run(ldb_engine.execute(debug=ENABLE_DEBUG))

    end = time()

    logger.info(f"Total execution time: {end - start} seconds")

