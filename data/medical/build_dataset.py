import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    dataset_path = Path(__file__).parent.parent.parent / "files/medical/data"
    ground_truth_path = Path(__file__).parent.parent.parent / "files/medical/raw_results/ground_truths"

    base_table_path = dataset_path / "patient_data.csv"
    image_skin_ref_path = dataset_path / "image_skin_data.csv"
    image_x_ray_ref_path = dataset_path / "image_x_ray_data.csv"
    text_symptoms_ref_path = dataset_path / "text_symptoms_data.csv"

    base_df = pd.read_csv(base_table_path).reset_index(drop=True)
    image_skin_df = pd.read_csv(image_skin_ref_path).reset_index(drop=True)
    image_x_ray_df = pd.read_csv(image_x_ray_ref_path).reset_index(drop=True)
    text_symptoms_df = pd.read_csv(text_symptoms_ref_path).reset_index(drop=True)

    merged_df = base_df.merge(
        image_skin_df,
        on="patient_id",
        how="left",
        suffixes=("", "_skin")
    ).merge(
        image_x_ray_df,
        on="patient_id",
        how="left",
        suffixes=("", "_xray")
    ).merge(
        text_symptoms_df,
        on="patient_id",
        how="left",
        suffixes=("", "_text")
    )

    merged_df.to_csv(Path(__file__).parent / "data_full.csv", index=False)

