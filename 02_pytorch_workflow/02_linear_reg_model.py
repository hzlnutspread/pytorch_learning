#!/usr/bin/env python3

import torch
from torch import nn
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else torch.device("cpu")

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


class LinearRegressionModel(
    nn.Module
):  # <- Almost everything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                1,
                requires_grad=True,
                dtype=torch.float,
            )
        )
        self.bias = nn.Parameter(
            torch.randn(
                1,
                requires_grad=True,
                dtype=torch.float,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- x is input data
        return self.weight * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(model_0)
print(model_0.state_dict())

with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)
print(y_test)
# plot_predictions(predictions=y_preds)

# loss function
loss_fn = nn.L1Loss()
print(loss_fn)
# optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)  # lr = learning rate
