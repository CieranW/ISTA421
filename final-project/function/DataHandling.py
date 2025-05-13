"""
This module handles the data processing for the cancer issue dataset.
It includes functions to read the data from a CSV file and split it into subsets based on cancer type.
The data is downloaded from Kaggle using the kagglehub library.
The data is then split into subsets based on the cancer type and returned as a dictionary of DataFrames.
"""

import pandas as pd
import numpy as np
import kagglehub as kh


def getData() -> pd.DataFrame:
    """
    This function reads the data from the CSV file and returns a DataFrame.
    """

    path = kh.dataset_download("preetigupta004/cancer-issue")
    df = pd.read_csv(f"{path}/cancer issue.csv")

    return df


def splitData(df: pd.DataFrame) -> dict:
    """
    This function splits the data into subsets based on cancer type and returns a dictionary of DataFrames.
    """
    # Split the data into subsets based on cancer type
    df_breast = df[df["CancerType"] == "breast"]
    df_colon = df[df["CancerType"] == "colon"]
    df_lung = df[df["CancerType"] == "lung"]
    df_prostate = df[df["CancerType"] == "prostate"]
    df_skin = df[df["CancerType"] == "skin"]

    # Create a dictionary of DataFrames
    data_dict = {
        "breast": df_breast,
        "colon": df_colon,
        "lung": df_lung,
        "prostate": df_prostate,
        "skin": df_skin,
    }

    return data_dict
