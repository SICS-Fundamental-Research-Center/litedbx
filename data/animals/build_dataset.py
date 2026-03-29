"""
Dataset construction script for animal data.

This script:
1. Groups the data by City
2. Concatenates other fields (ImagePath, Species, StationID) using comma
3. Validates that no field value contains a comma before concatenation
"""

import pandas as pd
from pathlib import Path


def check_no_commas(value, field_name, row_idx):
    """Check if a value contains a comma. Raise error if it does."""
    if pd.notna(value) and ',' in str(value) and field_name == 'ImagePath':
        raise ValueError(
            f"Error in row {row_idx}: Field '{field_name}' contains a comma. "
            f"Value: '{value}'. This would break the concatenated output."
        )


def concatenate_values(series):
    """
    Concatenate series values using comma.
    Filter out NaN values and duplicates.
    """
    # Filter out NaN values
    non_na_values = series.dropna()
    # Get unique values (maintain order)
    unique_values = non_na_values.unique()
    return ','.join(str(v) for v in unique_values)


def build_dataset(input_csv: Path, output_csv: Path):
    """
    Build the dataset by grouping by City and concatenating other fields.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
    """
    # Read the CSV file
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Update the image path.
    df["ImagePath"] = df["ImagePath"].apply(
        lambda x: x[21:] if pd.notna(x) else x)

    # Validate: Check that no field values (except City) contain commas
    print("\nValidating data...")
    for idx, row in df.iterrows():
        for col in df.columns:
            if col != 'City':
                check_no_commas(row[col], col, idx)

    print("Validation complete: No commas found in field values.")

    # Group by City and aggregate other fields
    print("\nGrouping by City and concatenating fields...")
    grouped_df = df.groupby('City').agg({
        'ImagePath': concatenate_values,
        'Species': concatenate_values,
        'StationID': concatenate_values
    }).reset_index()

    # Reorder columns to put City first
    grouped_df = grouped_df[['City', 'ImagePath', 'Species', 'StationID']]

    print(f"Grouped into {len(grouped_df)} unique cities.")

    # Write to output
    print(f"\nWriting output to {output_csv}...")
    grouped_df.to_csv(output_csv, index=False)

    print("Done!")
    print(f"\nSummary:")
    print(f"  Input rows: {len(df)}")
    print(f"  Output rows (unique cities): {len(grouped_df)}")
    print(f"  Cities: {', '.join(sorted(grouped_df['City'].tolist()))}")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent
    output_file = base_dir / "data_full.csv"

    dataset_path = Path(__file__).parent.parent.parent / "files/animals/source_data/sf_200/image_data.csv"

    # Build the dataset
    build_dataset(dataset_path, output_file)
