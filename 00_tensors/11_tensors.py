#!/usr/bin/env python3

import torch
import numpy as np

print("=" * 30)

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).type(torch.float32)

print(array, tensor)

print("=" * 30)

tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor)
