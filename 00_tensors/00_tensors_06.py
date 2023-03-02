#!/usr/bin/env python3

import torch

print("=" * 30)

tensor = torch.tensor([1, 2, 3])
print(tensor)

print("=" * 30)

print(tensor + 10)
print(torch.add(tensor, 10))

print("=" * 30)

print(tensor * 10)
print(torch.mul(tensor, 10))

print("=" * 30)
print(tensor - 10)
print("=" * 30)
print(tensor / 10)
