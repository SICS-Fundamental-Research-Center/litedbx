import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    repo_root = Path(__file__).parent.parent.parent
    dataset_path = repo_root / "files/medical/data"
    ground_truth_path = repo_root / "files/medical/raw_results/ground_truths"

    base_table_path = dataset_path / "patient_data.csv"
    image_skin_ref_path = dataset_path / "image_skin_data.csv"
    image_x_ray_ref_path = dataset_path / "image_x_ray_data.csv"
    text_symptoms_ref_path = dataset_path / "text_symptoms_data.csv"
    audio_ref_path = dataset_path / "audio_lung_data.csv"

    base_df = pd.read_csv(base_table_path).reset_index(drop=True)
    image_skin_df = pd.read_csv(image_skin_ref_path).reset_index(drop=True)
    image_x_ray_df = pd.read_csv(image_x_ray_ref_path).reset_index(drop=True)
    text_symptoms_df = pd.read_csv(text_symptoms_ref_path).reset_index(drop=True)
    audio_df = pd.read_csv(audio_ref_path).reset_index(drop=True)

    audio_df["path"] = audio_df["path"].map(lambda p: str(repo_root / p))
    audio_paths = (
        audio_df.groupby("patient_id")["path"].agg(";".join).rename("audio_path").reset_index()
    )

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
    ).merge(
        audio_paths,
        on="patient_id",
        how="left",
    )
    merged_df["audio_path"] = merged_df["audio_path"].fillna("")

    merged_df.to_csv(Path(__file__).parent / "data_full.csv", index=False)

