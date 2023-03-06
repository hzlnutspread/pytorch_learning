#!/usr/bin/env python3

import torch


zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros * torch.rand(3, 4))

print("=" * 30)

ones = torch.ones(
    size=(
        3,
        4,
    )
)
print(ones)
print(ones.dtype)
