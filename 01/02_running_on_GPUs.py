#!/usr/bin/env python3

import torch

print("=" * 30)

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built)
