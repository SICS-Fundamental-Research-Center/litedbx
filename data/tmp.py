import pandas as pd
import duckdb


if __name__ == "__main__":
    conn = duckdb.connect("databases/medical_database.duckdb")

    q1 = """
    select * from patients natural join SEM_FILTER_1;
    """

    q3 = """
    select * from patients natural join SEM_FILTER_2 where patients.did_family_have_cancer = 1;
    """

    q8 = """
    select * from patients natural join SEM_FILTER_3 where patients.did_family_have_cancer = 1;
    """

    df1 = conn.execute(q1).df()
    df3 = conn.execute(q3).df()
    df8 = conn.execute(q8).df()

    gt1 = pd.read_csv("src/evaluation/ground_truth/medical/Q1.csv")
    gt3 = pd.read_csv("src/evaluation/ground_truth/medical/Q3.csv")
    gt8 = pd.read_csv("src/evaluation/ground_truth/medical/Q8.csv")

    gt1_ids = set(gt1["patient_id"].tolist())
    gt3_ids = set(gt3["patient_id"].tolist())
    gt8_ids = set(gt8["patient_id"].tolist())

    df1["label"] = df1["patient_id"].apply(lambda x: 1 if x in gt1_ids else 0)
    df3["label"] = df3["patient_id"].apply(lambda x: 1 if x in gt3_ids else 0)
    df8["label"] = df8["patient_id"].apply(lambda x: 1 if x in gt8_ids else 0)

    idset1 = set(df1["patient_id"].tolist())
    idset3 = set(df3["patient_id"].tolist())
    idset8 = set(df8["patient_id"].tolist())

    assert gt1_ids.issubset(idset1)
    assert gt3_ids.issubset(idset3)
    assert gt8_ids.issubset(idset8)

    df1 = df1.drop(
        columns=["patient_id", "symptoms"]
    )
    df3 = df3.drop(
        columns=["patient_id", "image_path"]
    )
    df8 = df8.drop(
        columns=["patient_id", "image_path"]
    )

    df1.to_csv("tmp/medical_q1_complete.csv", index=False)
    df3.to_csv("tmp/medical_q3_complete.csv", index=False)
    df8.to_csv("tmp/medical_q8_complete.csv", index=False)


    conn.close()
