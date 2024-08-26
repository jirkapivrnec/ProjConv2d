import torch
import torch.nn as nn


class ProjConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ProjConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Learnable projection matrix
        self.projection_matrix = nn.Parameter(torch.randn(out_channels, in_channels * kernel_size * kernel_size))

    def forward(self, x):
        # Unfold the input into patches of shape (batch_size, num_patches, in_channels * kernel_size * kernel_size)
        x_unfolded = nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # Apply the projection to each patch
        projected = torch.matmul(self.projection_matrix, x_unfolded)
        # Reshape back to (batch_size, out_channels, new_height, new_width)
        batch_size, out_channels, num_patches = projected.size()
        new_height = int((x.size(2) + 2 * self.padding - self.kernel_size) / self.stride + 1)
        new_width = int((x.size(3) + 2 * self.padding - self.kernel_size) / self.stride + 1)
        projected = projected.view(batch_size, out_channels, new_height, new_width)
        return projected