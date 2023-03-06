#!/usr/bin/env python3

import torch

print("=" * 30)

x = torch.arange(0, 100, 10)
print(x)

min_x = torch.min(x)
print(min_x)

max_x = torch.max(x)
print(max_x)

mean_x = torch.mean(x.type(torch.float32))
print(mean_x)

sum_x = torch.sum(x)
print(sum_x)

arg_min = torch.argmin(x)
arg_max = torch.argmax(x)
print(arg_min)
print(arg_max)
