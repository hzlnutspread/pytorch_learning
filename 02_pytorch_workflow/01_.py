#!/usr/bin/env python3

import torch
from torch import nn
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else torch.device("cpu")

print("=" * 30)

print(torch.__version__)
