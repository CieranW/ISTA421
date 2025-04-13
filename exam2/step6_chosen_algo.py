import numpy as np
import pandas as pd


# -----------------------------------------------
# Define a Node data structure for the decision tree
# -----------------------------------------------
class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        """
        A Node in the decision tree.

        Parameters:
            feature_index: Index of the feature used for splitting.
            threshold: The threshold value for the split.
            left: Left child node.
            right: Right child node.
            value: Predicted value at a leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Only set for leaf nodes


# -----------------------------------------------
# Decision Tree Regressor (from scratch)
# -----------------------------------------------
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        """
        A basic decision tree regressor.

        Parameters:
            max_depth: The maximum depth of the tree.
            min_samples_split: Minimum number of samples required to split a node.
            n_features: Number of features to consider at each split. If None, uses all features.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # For feature bagging (random subset of features)
        self.root = None

    def fit(self, X, y):
        # Determine total number of features from data.
        self.n_features_all = X.shape[1]
        if self.n_features is None:
            # Use the square root of total features by default (common practice in RF).
            self.n_features = int(np.sqrt(self.n_features_all))
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features_all = X.shape

        # Stopping conditions: minimum samples or max depth reached.
        if n_samples < self.min_samples_split or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Randomly select a subset of features to consider for splitting (feature bagging).
        feature_indices = np.random.choice(
            n_features_all, self.n_features, replace=False
        )

        # Initialize parameters for finding the best split.
        best_feature, best_threshold = None, None
        best_gain = -np.inf
        best_splits = None

        current_impurity = self._variance(y)

        # Iterate over the randomly selected features.
        for feature_index in feature_indices:
            # Consider unique values in the feature as potential thresholds.
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Split data based on the threshold.
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # Skip invalid splits

                y_left = y[left_mask]
                y_right = y[right_mask]

                impurity_left = self._variance(y_left)
                impurity_right = self._variance(y_right)
                n_left, n_right = len(y_left), len(y_right)

                # Compute the variance gain (impurity reduction)
                gain = (
                    current_impurity
                    - (n_left / n_samples) * impurity_left
                    - (n_right / n_samples) * impurity_right
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
                    best_splits = {
                        "left_X": X[left_mask],
                        "left_y": y_left,
                        "right_X": X[right_mask],
                        "right_y": y_right,
                    }

        # If no valid split is found, return a leaf node.
        if best_gain == -np.inf:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Recursively build the left and right subtrees.
        left_subtree = self._build_tree(
            best_splits["left_X"], best_splits["left_y"], depth + 1
        )
        right_subtree = self._build_tree(
            best_splits["right_X"], best_splits["right_y"], depth + 1
        )
        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _calculate_leaf_value(self, y):
        """Calculate the prediction value for a leaf node (using the mean in regression)."""
        return np.mean(y)

    def _variance(self, y):
        """
        Calculate the total variance (sum of squared deviations) for the target values y.
        Multiplying by the number of samples gives a measure of total impurity.
        """
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)

    def predict(self, X):
        """Predict target values for samples in X."""
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions

    def _predict_sample(self, x, tree):
        """Recursively traverse the tree to get prediction for a single sample."""
        if tree.value is not None:
            return tree.value
        if x[tree.feature_index] <= tree.threshold:
            return self._predict_sample(x, tree.left)
        else:
            return self._predict_sample(x, tree.right)


# -----------------------------------------------
# Random Forest Regressor (from scratch)
# -----------------------------------------------
class RandomForestRegressor:
    def __init__(
        self, n_trees=10, max_depth=None, min_samples_split=2, n_features=None
    ):
        """
        Random Forest regressor using an ensemble of decision trees.

        Parameters:
            n_trees: Number of decision trees to build.
            max_depth: Maximum depth of each tree.
            min_samples_split: Minimum samples required to split a node.
            n_features: Number of features to consider when looking for the best split in each tree.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """Build a forest of decision trees from the training set (using bootstrap sampling)."""
        self.trees = []
        n_samples = X.shape[0]
        for i in range(self.n_trees):
            # Create a bootstrap sample (sample with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y.iloc[indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predict target values by averaging predictions from all trees."""
        # Collect predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Average predictions across trees
        return np.mean(tree_predictions, axis=0)
