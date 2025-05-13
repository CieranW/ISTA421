"""
This module looks at the data and performs exploratory data analysis (EDA) on the cancer issue dataset.
It includes functions to identify relationships using clustering, k-means, and PCA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    This function plots a correlation matrix for the DataFrame.
    It helps in understanding the relationships between different features in the dataset.
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def plot_pairwise_relationships(df: pd.DataFrame, hue: str) -> None:
    """
    This function plots pairwise relationships in the dataset using seaborn's pairplot.
    It helps in visualizing the relationships between different features and the target variable.
    """
    sns.pairplot(df, hue=hue)
    plt.title("Pairwise Relationships")
    plt.show()


def plot_pca(df: pd.DataFrame, n_components: int = 2) -> None:
    """
    This function performs PCA on the dataset and plots the first two principal components.
    It helps in visualizing the data in a lower-dimensional space.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title("PCA Result")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def plot_kmeans_clusters(df: pd.DataFrame, n_clusters: int = 3) -> None:
    """
    This function performs KMeans clustering on the dataset and plots the clusters.
    It helps in understanding the grouping of data points based on the features.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(scaled_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        scaled_data[:, 0],
        scaled_data[:, 1],
        c=kmeans.labels_,
        cmap="viridis",
        alpha=0.5,
    )
    plt.title("KMeans Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_silhouette_score(df: pd.DataFrame, n_clusters_range: List[int]) -> None:
    """
    This function calculates and plots the silhouette score for different numbers of clusters.
    It helps in determining the optimal number of clusters for KMeans clustering.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    silhouette_scores = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(scaled_data)
        silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, silhouette_scores, marker="o")
    plt.title("Silhouette Score for Different Numbers of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()


def plot_feature_importance(df: pd.DataFrame, target: str) -> None:
    """
    This function calculates and plots the feature importance using Random Forest.
    It helps in understanding the contribution of each feature to the target variable.
    """

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=X.columns)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, column: str) -> None:
    """
    This function plots the distribution of a categorical variable in the dataset.
    It helps in understanding the frequency of each category in the variable.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()


def plot_numerical_distribution(df: pd.DataFrame, column: str) -> None:
    """
    This function plots the distribution of a numerical variable in the dataset.
    It helps in understanding the distribution of values in the variable.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_histogram(df: pd.DataFrame, column: str) -> None:
    """
    This function plots a histogram for the specified column in the DataFrame.
    It helps in understanding the distribution of the data for that column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_boxplot(df: pd.DataFrame, column: str) -> None:
    """
    This function plots a box plot for the specified column in the DataFrame.
    It helps in identifying outliers and understanding the spread of the data.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Box Plot of {column}")
    plt.xlabel(column)
    plt.show()


def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str) -> None:
    """
    This function plots a scatter plot for the specified x and y columns in the DataFrame.
    It helps in understanding the relationship between the two variables.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_column], y=df[y_column])
    plt.title(f"Scatter Plot of {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
