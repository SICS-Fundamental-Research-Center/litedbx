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
    whitelist_ids = [1623,1624,5300,5303,5299,5301,5314,10037,10102,3312,3462,4182512799,2045,2606,2607,3479,43047,4811,4817,2048,4038,4800,4805]
    df_core = df[df['id'].isin(whitelist_ids)]
    budget = 5000 - len(df_core)
    df_rem = df.sample(n=budget, random_state=42)
    df = pd.concat([df_core, df_rem], ignore_index=True)

    df.to_csv("data_full.csv", index=False)

