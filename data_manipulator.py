import pandas as pd
from pathlib import Path
from typing import Optional
import asyncio
from llm_client import LiteLLMWrapper, BooleanFeatureResponse


class DataManipulator:
    """A class to manipulate pandas dataframes with semantic mapping capabilities."""

    def __init__(self, csv_path: str):
        """Initialize the DataManipulator by loading a CSV file.

        Args:
            csv_path: Path to the CSV file to load.
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df.reset_index(drop=True)
        self.llm_client = LiteLLMWrapper()

    def detect_modality(self, data_item: str) -> str:
        """Detect the modality of a data item.

        Args:
            data_item: The data item to analyze.

        Returns:
            The modality type ('TEXT' or 'IMAGE').
        """
        file_path = Path(data_item)
        if any(
            data_item.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]
        ):
            return "IMAGE"
        else:
            return "TEXT"

    def semantic_map(self, col_name: str, prompt: str, new_col_name: Optional[str] = None) -> None:
        """Apply semantic mapping to a column and append the result as a new column.

        Args:
            col_name: Name of the column to map.
            prompt: The prompt to use for semantic mapping.
            new_col_name: Name for the new column. If None, uses '{col_name}_mapped'.
        """
        if new_col_name is None:
            new_col_name = f"{col_name}_mapped"

        # Fetch the target column as data_items
        data_items = self.df[col_name].astype(str).tolist()

        # Peek the first data item to detect modality
        first_item = data_items[0] if data_items else ""
        modality = self.detect_modality(first_item)

        print(f"Detected modality: {modality}")
        print(f"Processing {len(data_items)} items...")

        # Call invoke_parallel_with_proxy
        results = asyncio.run(
            self.llm_client.invoke_parallel_with_proxy(
                modality=modality,
                prompt=prompt,
                data_items=data_items,
                response_model=BooleanFeatureResponse,
            )
        )

        # Append the mapped new column to the dataframe
        self.df[new_col_name] = results

        # Store the updated dataframe

        print(f"Semantic mapping completed. New column '{new_col_name}' added.")

    def store_dataframe(self, output_path: Optional[str] = None) -> None:
        """Store the dataframe to a CSV file.

        Args:
            output_path: Path to save the CSV file. If None, saves to 'output.csv'.
        """
        if output_path is None:
            output_path = "output.csv"
        self.df.to_csv(output_path, index=False)
        print(f"Dataframe saved to {output_path}")

    def evaluate_binary_classification(self, pred_col: str, true_col: str) -> dict:
        """Evaluate binary classification performance between two columns.

        Args:
            pred_col: Name of the column with predictions (0/1 or True/False).
            true_col: Name of the column with ground truth labels (0/1 or True/False).

        Returns:
            A dictionary containing TP, FP, TN, FN, precision, recall, and F1-score.

        Raises:
            ValueError: If columns contain values other than 0/1 or True/False.
        """
        # Validate that columns contain only binary values
        for col_name, col in [(pred_col, self.df[pred_col]), (true_col, self.df[true_col])]:
            # Get unique values
            unique_vals = set(col.dropna().unique())

            # Check if all values are valid binary (0/1 or True/False)
            valid_numeric = unique_vals.issubset({0, 1})
            valid_boolean = unique_vals.issubset({True, False})

            if not (valid_numeric or valid_boolean):
                raise ValueError(
                    f"Column '{col_name}' contains invalid values: {unique_vals}. "
                    f"Expected only 0/1 or True/False."
                )

        # Normalize both columns to boolean
        y_pred = pd.to_numeric(self.df[pred_col], errors='coerce').fillna(self.df[pred_col]).astype(bool)
        y_true = pd.to_numeric(self.df[true_col], errors='coerce').fillna(self.df[true_col]).astype(bool)

        # Calculate confusion matrix components
        tp = ((y_pred == True) & (y_true == True)).sum()
        tn = ((y_pred == False) & (y_true == False)).sum()
        fp = ((y_pred == True) & (y_true == False)).sum()
        fn = ((y_pred == False) & (y_true == True)).sum()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results = {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        # Print results
        print("\n=== Evaluation Results ===")
        print(f"True Positives (TP):  {results['true_positives']}")
        print(f"True Negatives (TN):  {results['true_negatives']}")
        print(f"False Positives (FP): {results['false_positives']}")
        print(f"False Negatives (FN): {results['false_negatives']}")
        print(f"\nPrecision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")

        return results


if __name__ == "__main__":
    # Example usage
    sem_map_prompt = f"""
    You are an allergy specialist analyzing patient symptoms to detect allergies. Please analyse the provided symptom. Return True if allergy is present, False otherwise. Do not return any explanations or additional text.
    """

    manipulator = DataManipulator("data/medical_q1_complete.csv")
    manipulator.semantic_map(
        col_name="symptoms",
        prompt=sem_map_prompt,
        new_col_name="LLM_label"
    )
    manipulator.store_dataframe("data/medical_q1_with_llm_labels.csv")

    dm = DataManipulator("data/medical_q1_with_llm_labels.csv")
    dm.evaluate_binary_classification(pred_col="LLM_label", true_col="label")
