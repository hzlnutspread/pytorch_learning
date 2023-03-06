#!/usr/bin/env python3

import torch

print("=" * 30)

x = torch.arange(1.0, 10.0)
print(x)
print(x.shape)

print("=" * 30)

x_reshaped = x.reshape(1, 9)
print(x_reshaped)

print("=" * 30)

x_view = x.view(1, 9)
print(x_view)

print("=" * 30)

x_hstack = torch.stack([x, x, x, x], dim=0)
print(x_hstack)
x_vstack = torch.stack([x, x, x, x], dim=1)
print(x_vstack)

print("=" * 30)

x_reshaped_squeezed = x_reshaped.squeeze()
# removes the single dimensions
print(x_reshaped)
print(x_reshaped_squeezed)

print("=" * 30)

x_reshaped_squeezed_unsqueezed = x_reshaped_squeezed.unsqueeze(dim=0)
# adds a single dimension
print(x_reshaped_squeezed_unsqueezed)

print("=" * 30)

x_original = torch.rand(size=(224, 224, 3))
x_original_permuted = x_original.permute(2, 0, 1)
print(x_original_permuted.shape)  # (3, 224, 224)
