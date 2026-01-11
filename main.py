from time import time
import asyncio
from workloads import medical_workloads


if __name__ == "__main__":

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
    print(f"Total execution time: {end - start} seconds")



