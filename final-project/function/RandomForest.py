"""
This module is the Random Forest classifier for the cancer issue dataset.
Model is built from scratch using the Random Forest algorithm.
It includes functions to train the model, make predictions, and evaluate the model's performance.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import multiprocessing as mp

# Helper function
def randomForest(X, y):
    """
    Train a Random Forest classifier on the given data.
    :param X: Feature matrix.
    :param y: Target vector.
    :return: Trained Random Forest model.
    """
    print("Random Forest Classifier")
    rf = RandomForest(n_trees=30, max_depth=8)
    rf.fitTree(X, y)
    accuracy = rf.score(X, y)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Feature Importances: {rf.feature_importances_}")
    

# GINI Impurity
def gini(y) -> float:
    """
    Calculate the Gini impurity for a list of labels.
    """
    labels, counts = np.unique(y, return_counts=True)
    prob_sq = (counts / counts.sum()) ** 2
    impurity = 1 - prob_sq.sum()
    return impurity


# Tree Node Class
class TreeNode:
    """
    A class representing a node in the decision tree.
    """

    def __init__(
        self,
        is_leaf=False,
        prediction=None,
        feature=None,
        threshold=None,
        branches=None,
    ):
        """
        Initialize a tree node.
        :param is_leaf: Boolean indicating if the node is a leaf node.
        :param prediction: Prediction value for leaf nodes.
        :param feature: Feature index for splitting.
        :param threshold: Threshold value for splitting.
        :param branches: Dictionary of branches for the node.
        """
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.branches = branches if branches is not None else {}
        self.left = None
        self.right = None


# Decision Tree Class
class DecisionTree:
    """
    A class representing a decision tree.
    """

    def __init__(self, max_depth=None, min_samples=2, min_gain=0.0001):
        """
        Initialize the decision tree.
        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum number of samples required to split a node.
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_gain = min_gain
        self.feature_importances_ = defaultdict(float)
        self.root = None

    def _fit(self, X, y, depth=0) -> None:
        """
        Fit the decision tree to the training data.
        :param X: Feature matrix.
        :param y: Target vector.
        :param depth: Current depth of the tree.
        """
        if len(y) == 0:
            return TreeNode(is_leaf=True, prediction=0)  # fallback

        if len(set(y)) == 1 or len(y) < self.min_samples or depth >= self.max_depth:
            return TreeNode(is_leaf=True, prediction=Counter(y).most_common(1)[0][0])

        n_samples, n_features = X.shape
        best_gain, best_feature, best_threshold, best_splits = 0, None, None, None

        for feature in range(n_features):
            values = np.unique(X[:, feature])

            if len(values) < 10 and X[:, feature].dtype.kind in {"i"}:
                splits = {val: y[X[:, feature] == val] for val in values}
                impurity = sum(
                    len(subset) / len(y) * gini(subset) for subset in splits.values()
                )
                gain = gini(y) - impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = None
                    best_splits = splits
            else:
                for val in values:
                    left = X[:, feature] <= val
                    right = ~left

                    if left.sum() == 0 or right.sum() == 0:
                        continue

                    left_y, right_y = y[left], y[right]

                    gain = gini(y) - (
                        (len(left_y) / len(y)) * gini(left_y)
                        + (len(right_y) / len(y)) * gini(right_y)
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = val
                        best_splits = (
                            left,
                            right,
                        )

        if best_gain < self.min_gain:
            return TreeNode(
                is_leaf=True,
                prediction=Counter(y).most_common(1)[0][0],
            )

        self.feature_importances_[best_feature] += best_gain

        if best_threshold is not None:
            node = TreeNode(
                feature=best_feature,
                threshold=best_threshold,
            )
            left_X, left_y = X[best_splits[0]], y[best_splits[0]]
            right_X, right_y = X[best_splits[1]], y[best_splits[1]]
            node.left = self._fit(left_X, left_y, depth + 1)
            node.right = self._fit(right_X, right_y, depth + 1)
            return node
        else:
            node = TreeNode(feature=best_feature)
            for val, subset in best_splits.items():
                node.branches[val] = self._fit(
                    X[X[:, best_feature] == val], subset, depth + 1
                )
            return node

    def _train(self, X, y) -> None:
        """
        Train the decision tree.
        :param X: Feature matrix.
        :param y: Target vector.
        """
        self.root = self._fit(X, y)

    def _predictOne(self, x, node) -> int:
        """
        Predict the class for a single sample.
        :param x: Feature vector.
        :param node: Current node in the tree.
        :return: Predicted class.
        """
        if node.is_leaf:
            return node.prediction

        if node.threshold is not None:
            if x[node.feature] <= node.threshold:
                return self._predictOne(
                    x, node.left if x[node.feature] <= node.threshold else node.right
                )
            else:
                return self._predictOne(
                    x, node.right if x[node.feature] > node.threshold else node.left
                )
        else:
            return self._predictOne(
                x,
                node.branches.get(
                    x[node.feature], TreeNode(is_leaf=True, prediction=0)
                ),
            )

    def _predict(self, X) -> np.ndarray:
        """
        Predict the classes for a set of samples.
        :param X: Feature matrix.
        :return: Predicted classes.
        """
        return np.array([self._predictOne(x, self.root) for x in X])

    def _printTree(self, node=None, depth=0) -> None:
        """
        Print the decision tree.
        :param node: Current node in the tree.
        :param depth: Current depth of the tree.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            print(f"{' ' * depth * 2}Leaf: {node.prediction}")
            return

        if node.threshold is not None:
            print(f"{' ' * depth * 2}Feature {node.feature} <= {node.threshold}")
            self._printTree(node.left, depth + 1)
            print(f"{' ' * depth * 2}Feature {node.feature} > {node.threshold}")
            self._printTree(node.right, depth + 1)
        else:
            for val, branch in node.branches.items():
                print(f"{' ' * depth * 2}Feature {node.feature} == {val}")
                self._printTree(branch, depth + 1)


# Random Forest Class
class RandomForest:
    """
    A class representing a random forest classifier.
    """

    def __init__(self, n_trees=10, max_depth=10, min_samples=5):
        """
        Initialize the random forest.
        :param n_trees: Number of trees in the forest.
        :param max_depth: Maximum depth of each tree.
        :param min_samples: Minimum number of samples required to split a node.
        """
        self.n_trees = n_trees
        self.trees = []
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_importances_ = defaultdict(float)

    def _trainTree(self, seed, X, y) -> DecisionTree:
        """
        Train a single decision tree.
        :param seed: Random seed for reproducibility.
        :param X: Feature matrix.
        :param y: Target vector.
        :return: Trained decision tree.
        """
        np.random.seed(seed)
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
        tree._train(X_sample, y_sample)
        return tree

    def fitTree(self, X, y) -> None:
        """
        Fit the random forest to the training data.
        :param X: Feature matrix.
        :param y: Target vector.
        """

        seeds = np.random.randint(0, 1000000, self.n_trees)

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(self._trainTree, [(seed, X, y) for seed in seeds])
        self.trees = results

        for tree in self.trees:
            for feature, importance in tree.feature_importances_.items():
                self.feature_importances_[feature] += importance / len(self.trees)

    def _predict(self, X) -> np.ndarray:
        """
        Predict the classes for a set of samples.
        :param X: Feature matrix.
        :return: Predicted classes.
        """
        predictions = np.array([tree._predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda row: Counter(row).most_common(1)[0][0], axis=0, arr=predictions
        )

    def _featureImportance(self) -> None:
        """
        Calculate the feature importances for the random forest.
        """
        for tree in self.trees:
            for feature, importance in tree.feature_importances_.items():
                self.feature_importances_[feature] += importance / len(self.trees)
        return self.feature_importances_

    def score(self, X, y) -> float:
        """
        Calculate the accuracy of the random forest.
        :param X: Feature matrix.
        :param y: Target vector.
        :return: Accuracy of the random forest.
        """
        predictions = self._predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy
