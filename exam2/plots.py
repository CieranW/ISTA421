import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_data(y, predictions):
    """
    Plot the actual vs predicted values.

    Parameters:
        y: Actual values.
        predictions: Predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.grid()
    plt.show()
    # Plotting the residuals
    residuals = y - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.grid()
    plt.show()
    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
    # Box plot of residuals
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=residuals)
    plt.title("Box Plot of Residuals")
    plt.xlabel("Residuals")
    plt.grid()
    plt.show()
