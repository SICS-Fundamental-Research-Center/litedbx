import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    dataset_path = Path(__file__).parent.parent.parent / "files/movie/data"
    ground_truth_path = Path(__file__).parent.parent.parent / "files/movie/raw_results/ground_truths"

    base_table_path = dataset_path / "rotten_tomatoes_movies.csv"
    review_table_path = dataset_path / "rotten_tomatoes_movie_reviews.csv"

    base_df = pd.read_csv(base_table_path).reset_index(drop=True)
    review_df = pd.read_csv(review_table_path).reset_index(drop=True)

    merged_df = base_df.merge(
        review_df,
        on="id",
        how="left",
        suffixes=("", "_review")
    )
    merged_df.to_csv(Path(__file__).parent / "data_full.csv", index=False)

