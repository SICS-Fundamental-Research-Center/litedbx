from time import time
import logging
import sys
from workloads import medical
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

    queries = [
        "Q1", 
        # "Q3", 
        # "Q8"
    ]
    workload = medical.get_workload(queries=queries)

    ldb_engine = LdbEngine(workload)

    start = time()

    asyncio.run(ldb_engine.execute(debug=ENABLE_DEBUG))

    end = time()

    logger.info(f"Total execution time: {end - start} seconds")

