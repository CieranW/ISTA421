from step3_assessment import get_data
from step6_chosen_algo import RandomForestRegressor
from step7_chosen_validation import cross_validate
import pandas as pd


def main():
    # Get dataset
    df = get_data()

    X = df.drop(columns="Excess Readmission Ratio")
    y = df["Excess Readmission Ratio"]
    # Convert to NumPy arrays if starting from a Pandas DataFrame:
    X = X.values
    y = pd.Series(y)

    rf = RandomForestRegressor(n_trees=10, max_depth=5, min_samples_split=2)
    rf.fit(X, y)
    predictions = rf.predict(X)
    print("Predictions:", predictions[:5])  # Print first 5 predictions

    # Cross-validation
    mse = cross_validate(rf, X, y, n_splits=5)
    # Evaluate the model
    print("Cross-Validation MSE Scores:", mse)
    # Calculate average MSE
    avg_mse = sum(mse) / len(mse)
    print("Average MSE:", avg_mse)


if __name__ == "__main__":
    main()
