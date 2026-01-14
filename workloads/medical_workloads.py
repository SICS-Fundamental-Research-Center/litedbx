from ldb_engine import LDBEngine
from data_structures import UCQ, CQ



def q1():
    return UCQ(
        select_cols=["patient_id"],
        rules=[
            CQ(
                static_rules=[],
                sem_rules=[
                    (
                        "symptoms", 
                        "The patient has an allergy",
                        (
                            "You are a medical expert. " 
                            "Please determine if the given {COL} indicate: {CONDITION}. "
                            "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                            "Do NOT provide any explanations."
                        )
                    )
                ],
            ),
        ],
    )



WORKLOADS = {
    "Q1": q1(),
}



def build_query_engine(workloads, feature_enrich_budget=3, query_rewrite_budget=3):
    return LDBEngine(
        dataset_name="medical",
        workloads=_retrieve_workloads(workloads),
        feature_enrich_budget=feature_enrich_budget,
        query_rewrite_budget=query_rewrite_budget,
        external_keys=["image_path","skin_image_id","image_path_xray",
                       "xray_id","symptoms","symptom_id"]
    )


def _retrieve_workloads(workloads):
    return {name: WORKLOADS[name] for name in workloads}


