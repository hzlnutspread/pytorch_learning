#!/usr/bin/env python3

import torch

print("=" * 30)

tensor = torch.tensor([1, 2, 3])
mul_value = torch.matmul(tensor, tensor)

print("=" * 30)

tensor_A = torch.tensor(
    [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
)
tensor_B = torch.tensor(
    [
        [7, 10],
        [8, 11],
        [9, 12],
    ]
)

tensor_B = tensor_B.T

tensor_multiplied = torch.mm(tensor_A, tensor_B)
print(tensor_multiplied)
