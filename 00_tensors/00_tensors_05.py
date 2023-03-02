#!/usr/bin/env python3

import torch


print("=" * 30)

float_32_tensor = torch.tensor(
    [3.0, 6.0, 9.0], dtype=torch.float32, device="cpu", requires_grad=False
)
print(float_32_tensor)
print(float_32_tensor.dtype)

print("=" * 30)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)
print(float_16_tensor.dtype)

print("=" * 30)

print(float_16_tensor * float_32_tensor)

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)

print(float_32_tensor * int_32_tensor)
