#!/usr/bin/env python3

import torch


print("=" * 30)

range = torch.arange(start=0, end=11, step=1)
print(range)

print("=" * 30)

ten_zeros = torch.zeros_like(input=range)
print(ten_zeros)

print("=" * 30)
