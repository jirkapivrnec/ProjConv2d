import torch
import torch.nn as nn

from extendedCNN import SlidingProjectionNet

# Instantiate the model
model = SlidingProjectionNet()

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(total_params)