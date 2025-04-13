import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_data():
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv("FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv")

    # Summarize the dataset
    data_summ(df)

    # Clean Dataset
    clean_dataset(df)

    # Testing purposes
    # data_summ(df)
    # plot_data(df)
    # print(df["Number of Readmissions"].head())

    return df


def data_summ(df):
    """
    Summarize the dataset.
    """
    print("Dataset Summary:")
    print("Column names:", df.columns.tolist())
    print("Data types:")
    print(df.dtypes)
    print(df.describe())
    print(df.info())
    print(df.head())

    # Coefficient of variation
    print("Coefficient of Variation:")
    for col in df.columns:
        if df[col].dtype == "float64":
            mean = df[col].mean()
            std = df[col].std()
            cv = std / mean
            print(f"{col}: {cv:.2f}")


def clean_dataset(df):
    """
    Clean the dataset.
    """
    # List of columns to change from nan to 0
    cols_to_zero = [
        "Number of Readmissions",
        "Number of Discharges",
        "Footnote",
        "Excess Readmission Ratio",
        "Predicted Readmission Rate",
        "Expected Readmission Rate",
    ]
    # Turn N/A values into NaN
    df.replace("N/A", np.nan, inplace=True)
    # Turn empty strings into NaN
    df.replace("", np.nan, inplace=True)

    # Convert columns to numeric
    for col in cols_to_zero:
        nan_to_zero(df, col)


def nan_to_zero(df, col_name):
    """
    Turn all NaN values in the dataset into 0.
    """
    # Check data type of column
    if df[col_name].dtype == "float64":
        # Convert NaN to 0
        df[col_name] = df[col_name].fillna(0)
    elif df[col_name].dtype == "object":
        # Convert NaN to 0
        df[col_name] = (
            pd.to_numeric(df[col_name], errors="coerce").fillna(0).astype(int)
        )
    else:
        print(f"Warning: {col_name} column is not numeric. Skipping conversion.")
        return df
    return df


def plot_data(df):
    """
    Plot the dataset.
    """
    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create a bar plot for the number of readmissions
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Number of Readmissions", y="Facility Name", data=df)
    plt.title("Number of Readmissions by Provider")
    plt.xlabel("Number of Readmissions")
    plt.ylabel("Facility Name")
    plt.show()
    # Create a histogram for the excess readmission ratio
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Excess Readmission Ratio"], bins=30, kde=True)
    plt.title("Distribution of Excess Readmission Ratio")
    plt.xlabel("Excess Readmission Ratio")
    plt.ylabel("Frequency")
    plt.show()
    # Create a box plot for the predicted readmission rate
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Predicted Readmission Rate", data=df)
    plt.title("Box Plot of Predicted Readmission Rate")
    plt.xlabel("Predicted Readmission Rate")
    plt.show()
    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
