#!/usr/bin/env python3

import torch

print("=" * 30)

# prints a random 3 x 4 matrix
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)

print("=" * 30)

random_image_size_tensore = torch.rand(
    size=(224, 224, 3)
)  # height, width, colour channels (RGB)
print(random_image_size_tensore.shape, "\n", random_image_size_tensore.ndim)

print("=" * 30)

my_random_tensor = torch.rand(3, 4, 5)
print(my_random_tensor)
