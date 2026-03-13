import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    dataset_path = Path(__file__).parent.parent.parent / \
        "files/ecomm/data/fashion_product_images/styles_details.parquet"

    df = pd.read_parquet(dataset_path)

    # Extract typeName from nested category fields
    # (1) Extract from masterCategory
    df['masterCategory'] = df['masterCategory'].apply(lambda x: x['typeName'] if isinstance(x, dict) else x)

    # (2) Extract from subCategory
    df['subCategory'] = df['subCategory'].apply(lambda x: x['typeName'] if isinstance(x, dict) else x)

    # (3) Extract from articleType
    df['articleType'] = df['articleType'].apply(lambda x: x['typeName'] if isinstance(x, dict) else x)

    # (4) Map id to image_path
    # Get the base path from the dataset_path
    base_path = dataset_path.parent.parent.parent / "source_data/fashion-dataset/images"
    df['image_path'] = df['id'].apply(lambda x: str(base_path / f"{x}.jpg"))
    print(f"len(df) = {len(df)} (before existing filtering)")

    # Check if the referenced images exist and filter accordingly
    df = df[df['image_path'].apply(lambda x: Path(x).exists())]
    print(f"len(df) = {len(df)} (after existing filtering)")


    # (5) Get the combined column of {productDisplayName} {productDescriptors}.
    # Convert dictionaries to strings if needed
    df['productDisplayName'] = df['productDisplayName'].apply(
        lambda x: str(x) if isinstance(x, dict) else x
    )
    df['productDescriptors'] = df['productDescriptors'].apply(
        lambda x: str(x) if isinstance(x, dict) else x
    )

    df['productNameAndDesc'] = (
        "Product display Name: " + df['productDisplayName'] + "\n" +
        "---\n" +
        "Product Descriptors: " + df['productDescriptors']
    )


    # Sample the dataset to 10K rows for faster eval.
    # Keep the id within the whitelist to ensure the ground truth is held in the sampled data.
    whitelist_ids = [
        5299, 5300, 5301, 1623, 1624, 5303, 5314,  # Q1
        10037, 10102, 3312, 41825, 3462,  # Q2
        3351, 30292, 10689, 8419,  # Q7
        12799, 2048, 2606, 2607, 3479, 4038, 4800, 4805, 4817, 2045, 43047, 4811,  # Q8
        6241, 1891, 53126, 1563, 15779, 47525,  # Q9
        6100, 7935, 10579,  # Q10
        8103, 13112, 8402, 3470,  # 11
        43047, 12799, 4811,  # Q13
        18345, 29202,  # Q14
    ]
    df_core = df[df['id'].isin(whitelist_ids)]

    scale_factor = 2000
    print(f"len(whitelist_ids)={len(whitelist_ids)}, len(df_core)={len(df_core)}, len(set(whitelist_ids))={len(set(whitelist_ids))}")

    df_rem = df.sample(n=scale_factor - len(df_core), random_state=42)
    df_final = pd.concat([df_core, df_rem], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    print(len(df_core), len(df_rem), len(df_final))

    df_final.to_csv("data_full.csv", index=False)

