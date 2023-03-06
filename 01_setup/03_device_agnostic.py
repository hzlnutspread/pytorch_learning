#!/usr/bin/env python3

import torch

device = "mps" if torch.backends.mps.is_available() else torch.device("cpu")

print("=" * 30)

tensor = torch.tensor([1, 2, 3])
tensor_on_mps = tensor.to(device)
print(tensor_on_mps)
