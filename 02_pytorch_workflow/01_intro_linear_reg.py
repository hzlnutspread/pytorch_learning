#!/usr/bin/env python3

import torch
from torch import nn
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else torch.device("cpu")

print("=" * 30)

# parameters
weight = 0.7  # gradient
bias = 0.3  # intercept

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# creating a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


# Train model on training data
# Get model to predict y_test based on x_test data
# Compare how good predictions were based on actual y_test data
def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    plt.figure(figsize=(10, 7))
    # plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    # plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


plot_predictions()
