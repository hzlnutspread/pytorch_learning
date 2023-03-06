#!/usr/bin/env python3

import torch

print("=" * 30)

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS NOT AVAILABLE. PYTORCH NOT BUILT WITH MPS AVAILABLE")

    else:
        print("MPS NOT AVAILABLE. CURRENT MACOS IS NOT 12.3+")

else:
    mps_device = torch.device("mps")

# =============================================================================

print(mps_device)
print(torch.cuda.device_count)
