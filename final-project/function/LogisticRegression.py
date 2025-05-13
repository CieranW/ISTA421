import numpy as np


class LogisticRegressionScratch:
    """
    Logistic Regression Classifier using Scratch Implementation.
    This class implements the logistic regression algorithm from scratch.
    It includes methods for training the model, making predictions, and computing loss.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the Logistic Regression model.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z) -> np.ndarray:
        """
        Compute the sigmoid function.
        :param z: Input value.
        :return: Sigmoid of the input value.
        """
        # Sigmoid function
        return 1 / (1 + np.exp(-z))

    def _computeLoss(self, y_true, y_pred) -> float:
        """
        Compute the binary cross-entropy loss.
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Binary cross-entropy loss.
        """
        # Binary cross-entropy loss
        # Adding epsilon to avoid log(0)
        epsilon = 1e-15
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def _feedForward(self, X) -> np.ndarray:
        """
        Compute the feed-forward step.
        :param X: Input features.
        :return: Predicted probabilities.
        """
        # Feed-forward step
        z = np.dot(X, self.weights) + self.bias
        a = self._sigmoid(z)
        return a

    def fit(self, X, y) -> None:
        """
        Train the Logistic Regression model.
        :param X: Input features.
        :param y: True labels.
        """
        # Fit the model to the data
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            A = self._feedForward(X)
            self.losses.append(self._computeLoss(y, A))
            dz = A - y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained model.
        :param X: Input features.
        :return: Predicted labels.
        """
        threshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(y_hat)
        y_predicted_cls = (y_pred >= threshold).astype(int)

        return np.array(y_predicted_cls)

    def accuracy(self, y_true, y_pred) -> float:
        """
        Compute the accuracy of the model.
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Accuracy of the model.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

# Helper function to call logistic regression
def logisticRegression(X_train, y_train, X_test, y_test) -> tuple:
    """
    Helper function to call logistic regression.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Testing features.
    :param y_test: Testing labels.
    :return: Trained model and predictions.
    """
    print("Logistic Regression")
    model = LogisticRegressionScratch(learning_rate=0.01, num_iterations=10000)

    # Testing the model with some debug prints
    # print("X_train dtype:", X_train.dtype)
    # print("Any None in X_train?", np.any(X_train == None))
    # print("X_train example row:", X_train[0])
    model.fit(X_train, y_train)
    # Debug prints to check the weights and predictions
    # print("Weights after training:", model.weights)
    preds = model.predict(X_test)
    accuracy = model.accuracy(y_test, preds)
    print("Test Accuracy:", accuracy)
    print(f"Predictions: {preds}\n")
    return model, preds

# Helper function to split data into training and testing sets
def trainTestSplit(X, y, test_size=0.2, seed=None) -> tuple:
    """
    Split the data into training and testing sets.
    :param X: Input features.
    :param y: True labels.
    :param test_size: Proportion of the dataset to include in the test split.
    :param seed: Random seed for reproducibility.
    :return: Training and testing sets.
    """
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]