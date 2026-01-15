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


def q3():
    return UCQ(
        select_cols=["patient_id"],
        rules=[
            CQ(
                static_rules=[("did_family_have_cancer", "Eq", 1)],
                sem_rules=[
                    (
                        "image_path_xray", 
                        "This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image",
                        (
                            "You are a radiology expert. " 
                            "Please determine if the given {COL} indicate: {CONDITION}. "
                            "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                            "Do NOT provide any explanations."
                        )
                    )
                ],
            ),
        ],
    )


def q8():
    return UCQ(
        select_cols=["patient_id"],
        rules=[
            CQ(
                static_rules=[("did_family_have_cancer", "Eq", 1)],
                sem_rules=[
                    (
                        "image_path", 
                        "This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image",
                        (
                            "You are a dermatology expert. " 
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
    "Q3": q3(),
    "Q8": q8(),
}


def _retrieve_workloads(workloads):
    return {name: WORKLOADS[name] for name in workloads}


def build_query_engine(workloads, feature_enrich_budget=3, query_rewrite_budget=3, hitl_budget=50):
    return LDBEngine(
        dataset_name="medical",
        workloads=_retrieve_workloads(workloads),
        feature_enrich_budget=feature_enrich_budget,
        query_rewrite_budget=query_rewrite_budget,
        hitl_budget=hitl_budget,
        external_keys=["image_path","skin_image_id","image_path_xray",
                       "xray_id","symptoms","symptom_id"]
    )

