#!/usr/bin/env python3

import torch

print("=" * 30)

x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)

print("=" * 30)

print(x[0])
print(x[0][2][0])

# get all values of 0 dim but only the 1 index value of the 1st and 2nd dims
print(x[:, 1, 1])

# get index 0 of 0th and 1st dim and all values of 2nd dim
print(x[0, 0, :])

# return tensor([9])
print(x[:, 2, 2])

# return tensor ([3, 6 ,9])
print(x[:, :, 2])
