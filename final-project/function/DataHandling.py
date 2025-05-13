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


def initData(df) -> None:
    """
    This fuction gets the initial statistics of the data.
    It includes the shape of the data, the number of missing values, and the data types.
    """
    # Get the shape of the data
    print("Shape of the data:", df.shape)

    # Get summary statistics
    print("Summary statistics:\n", df.describe())
    print(df.info())

    # Get the number of unique values in each column
    print("Number of unique values in each column:\n", df.nunique())

    # Get the number of missing values
    print("Number of missing values:", df.isnull().sum().sum())

    # Get the data types
    print("Data types:\n", df.dtypes)

    # Get the first 5 rows of the data
    print("First 5 rows:\n", df.head())


# Had to move this here as it would not work if left in the other file
def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the data by removing unnecessary columns and handling missing values.
    It returns a cleaned DataFrame.
    """
    df = df.copy()

    # Turn Stage into numerical values
    df["Stage"] = df["Stage"].map(
        {
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
        }
    )

    # Convert Male/Female to 0/1
    df["Gender"] = df["Gender"].map(
        {
            "Male": 0,
            "Female": 1,
        }
    )

    # Convert Yes/No to 0/1
    df["FamilyHistory"] = df["FamilyHistory"].map(
        {
            "Yes": 1,
            "No": 0,
        }
    )
    df["Recurrence"] = df["Recurrence"].map(
        {
            "Yes": 1,
            "No": 0,
        }
    )

    # Convert TreatmentResponse to numerical values
    df["TreatmentResponse"] = df["TreatmentResponse"].map(
        {
            "Complete Remission": 1,
            "Partial Remission": 2,
            "No Response": 3,
        }
    )

    # Convert Ethnicity to numerical values
    df["Race/Ethnicity"] = df["Race/Ethnicity"].map(
        {
            "Caucasian": 1,
            "African American": 2,
            "Hispanic": 3,
            "Asian": 4,
            "Other": 5,
        }
    )

    # Convert SmokingStatus to numerical values
    df["SmokingStatus"] = df["SmokingStatus"].map(
        {
            "Non-Smoker": 1,
            "Former Smoker": 2,
            "Smoker": 3,
        }
    )

    # Convert TreatmentType to numerical values
    df["TreatmentType"] = df["TreatmentType"].map(
        {
            "Chemotherapy": 1,
            "Radiation": 2,
            "Surgery": 3,
            "Combination Therapy": 4,
        }
    )

    # Convert GeneticMarker to numerical values
    # Nan
    df["GeneticMarker"] = df["GeneticMarker"].fillna("None")
    df["GeneticMarker"] = df["GeneticMarker"].map(
        {
            "None": 0,
            "BRCA1": 1,
            "KRAS": 2,
            "EGFR": 3,
        }
    )

    # Convert CancerType to numerical values
    df["CancerType"] = df["CancerType"].map(
        {
            "Breast": 1,
            "Colon": 2,
            "Prostate": 3,
            "Lung": 4,
            "Leukemia": 5,
            "Skin": 6,
        }
    )

    
    # Drop unnecessary columns
    df.drop(columns=["HospitalRegion"], inplace=True)

    return df


def splitData(df: pd.DataFrame) -> dict:
    """
    This function splits the data into subsets based on cancer type and returns a dictionary of DataFrames.
    """
    data_dict = {}
    cancer_types = {
        1: "breast",
        2: "colon",
        3: "prostate",
        4: "lung",
        5: "leukemia",
        6: "skin"
    }

    for code, label in cancer_types.items():
        data_dict[label] = df[df["CancerType"] == code]

    return data_dict
