#!/usr/bin/env python3

import torch

device = "mps" if torch.backends.mps.is_available() else torch.device("cpu")

print("=" * 30)

tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)

tensor_on_mps = tensor.to(device)
print(tensor_on_mps)

tensor_back_on_cpu = tensor_on_mps.cpu().numpy()
print(tensor_back_on_cpu)
