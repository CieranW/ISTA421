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

    def _compute_loss(self, y_true, y_pred) -> float:
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

    def _feed_forward(self, X) -> np.ndarray:
        """
        Compute the feed-forward step.
        :param X: Input features.
        :return: Predicted probabilities.
        """
        # Feed-forward step
        z = np.dot(X, self.weights) + self.bias
        a = self._sigmoid(z)
        return a

    def _fit(self, X, y) -> None:
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
            A = self._feed_forward(X)
            self.losses.append(self._compute_loss(y, A))
            dz = A - y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained model.
        :param X: Input features.
        :return: Predicted labels.
        """
        threshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_pred]

        return np.array(y_predicted_cls)
