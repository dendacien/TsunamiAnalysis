import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

def retrieve_earthquake_data():
    file_path = "earthquake_data_tsunami.csv"
    print("Retrieving earthquake data from Kaggle dataset...")

    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset",
    file_path
    )

    # Check if the DataFrame from Kaggle is empty
    if df is None or df.empty:
        raise RuntimeError("Failed to retrieve earthquake data from Kaggle.")
    # Return the DataFrame
    return df
