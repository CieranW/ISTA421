from step3_assessment import get_data
from step6_chosen_algo import RandomForestRegressor
from step7_chosen_validation import cross_validate
from plots import plot_data
import pandas as pd
import matplotlib.pyplot as plt


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

    # Plotting the results
    # plot_data(y, predictions)

    # Feature importance plot
    try:
        feature_importances = rf.feature_importances_
        feature_names = df.drop(columns="Excess Readmission Ratio").columns
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )
        print("Feature Importances:")
        print(feature_importance_df)
        # Plot feature importances
        feature_importance_df.plot(
            kind="bar", x="Feature", y="Importance", title="Feature Importances"
        )
        plt.show()
    except AttributeError:
        print(
            "The RandomForestRegressor implementation does not support feature importances."
        )


if __name__ == "__main__":
    main()
