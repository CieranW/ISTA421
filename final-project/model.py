# Import functions from all the necessary modules
from function.DataHandling import (
    getData,
    initData,
    cleanData,
    splitData,
)
from function.DataExploration import (
    plotCorrelationMatrix,
    plotPairwiseRelationships,
    plotPCA,
    plotKmeansClusters,
    plotSilhouetteScore,
    plotFeatureImportance,
    plotCategoricalDistribution,
    plotNumericalDistribution,
    plotHistogram,
    plotBoxplot,
    plotScatter,
)
from function.LogisticRegression import logisticRegression, trainTestSplit
from function.RandomForest import randomForest
from multiprocessing import freeze_support

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # Get and clean data
    df = getData()
    clean_df = cleanData(df)
    df_dict = splitData(clean_df)

    # Logistic Regression
    feature = ["Age", "TumorSize", "BMI", "Stage", "CancerType", "TreatmentType", "GeneticMarker"]
    clean_df["SurvivalBinary"] = clean_df["SurvivalMonths"].apply(lambda x: 1 if x > 24 else 0)
    X_lr = clean_df[feature].values
    y_lr = clean_df["SurvivalBinary"].values
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = trainTestSplit(X_lr, y_lr, test_size=0.2, seed=42)

    lrModel, lrPredictions = logisticRegression(X_train_lr, y_train_lr, X_test_lr, y_test_lr)

    # Random Forest - Whole Dataset
    x_rf = clean_df[feature].values
    y_rf = clean_df["TreatmentResponse"].values
    randomForest(x_rf, y_rf)

    # Random Forest - By Cancer Type
    for cancer_type, df in df_dict.items():
        if len(df) == 0 or df["TreatmentResponse"].isnull().all():
            print(f"Skipping {cancer_type} cancer (no data)")
            continue

        if len(df["TreatmentResponse"].unique()) < 2:
            print(f"Skipping {cancer_type} cancer (only one class present)")
            continue
        
        print(f"\nRandom Forest for {cancer_type} cancer")
        x_rf = df[feature].values
        y_rf = df["TreatmentResponse"].values
        randomForest(x_rf, y_rf)