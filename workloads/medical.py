from pathlib import Path
from typing import Optional
import yaml
from data_structure import Predicate, SemPredicate, SemCQ
from .ldb_workload import LdbWorkload

DATASET_PATH = Path(__file__).parent.parent / "data/medical"
CURRENT_DIR = Path(__file__).parent

Q1 = SemCQ(
    selected=["patient_id"],
    Sigma=[Predicate("symptoms", "!=", "")],
    Ps=[
        SemPredicate(
            field="symptoms",
            modality="Text",
            succ_cond="The patient has an allergy",
            prompt=(
                "You are a medical expert. " 
                "Please determine if the given symptom indicate "
                "that the patient has an allergy. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])

Q3 = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("did_family_have_cancer", "==", 1),
        Predicate("image_path_xray", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path_xray",
            modality="Image",
            succ_cond="This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image",
            prompt=(
                "You are a radiology expert. " 
                "Please determine if the given X-ray image indicate "
                "that there are lung problems (considered sick/disease) according to the X-ray image. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])

Q8 = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("did_family_have_cancer", "==", 1),
        Predicate("image_path", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="Image",
            succ_cond="This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image",
            prompt=(
                "You are a dermatology expert. " 
                "Please determine if the given skin image indicate "
                "that it shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])

Q9 = SemCQ(
    selected=["patient_id"],
    Sigma=[
        Predicate("image_path", "!=", ""),
        Predicate("image_path_xray", "!=", ""),
    ],
    Ps=[
        SemPredicate(
            field="image_path",
            modality="Image",
            succ_cond="This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image",
            prompt=(
                "You are a dermatology expert. " 
                "Please determine if the given skin image indicate "
                "that it shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        )),
        SemPredicate(
            field="image_path_xray",
            modality="Image",
            succ_cond="This X-ray image shows a sick human lung according to the X-ray image",
            prompt=(
                "You are a radiology expert. " 
                "Please determine if the given X-ray image indicate "
                "that there are lung problems (considered sick/disease) according to the X-ray image. "
                "Please JUST answer \"True\" if they do, and \"False\" otherwise. "
                "Do NOT provide any explanations."
        ))
])


SEM_QUERIES = {
    "Q1": Q1,
    "Q3": Q3,
    "Q8": Q8,
    "Q9": Q9,
}


def get_workload(queries: list[str], config: Optional[dict] = None) -> LdbWorkload:
    sem_queries = {}
    for q in queries:
        assert q in SEM_QUERIES, f"Invalid query {q} in medical dataset."
        sem_queries[q] = SEM_QUERIES[q]
    
    if config is None:
        with open(CURRENT_DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
    assert config is not None, "Fail to load the workload config."

    return LdbWorkload(data_dir=str(DATASET_PATH), scenario="medical", queries=sem_queries, config=config)
