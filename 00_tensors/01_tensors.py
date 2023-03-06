#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import matplotlib as plt
import torchvision

scalar = torch.tensor(7)
print(scalar)
print(scalar.item())

print("=" * 30)

vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)

print("=" * 30)

MATRIX = torch.tensor(
    [
        [7, 8],
        [9, 10],
    ]
)
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

print("=" * 30)

TENSOR = torch.tensor(
    [
        [
            [1, 2, 3],
            [3, 6, 9],
            [2, 4, 5],
        ]
    ]
)
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)
