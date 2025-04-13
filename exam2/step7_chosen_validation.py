import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from step6_chosen_algo import RandomForestRegressor


def cross_validate(model, X, y, n_splits=5):
    """
    Perform K-Fold Cross Validation on the given model.

    Parameters:
    - model: The model to be evaluated.
    - X: Feature matrix.
    - y: Target variable.
    - n_splits: Number of folds for cross-validation.

    Returns:
    - mse_scores: List of mean squared error scores for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Instantiate and train the model
        rf = RandomForestRegressor(n_trees=10, max_depth=5, min_samples_split=2)
        rf.fit(X_train, y_train)

        # Predict and evaluate
        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_scores.append(mse)
        print(f"Fold {fold} MSE: {mse:.4f}")

    return mse_scores
