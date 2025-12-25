import torch
import torch.nn as nn
import numpy as np

class HighFrequencyExtractor(nn.Module):
    """
    Extracts high-frequency features (edges) using a fixed Sobel filter.
    This module is NOT trained.
    """
    def __init__(self, in_channels=3):
        super(HighFrequencyExtractor, self).__init__()
        self.in_channels = in_channels
        
        # Define Sobel filters
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Stack them to create a (out_channels=2, in_channels=1, 3, 3) weight
        sobel_weights_xy = np.stack([sobel_x, sobel_y], axis=0) # Shape (2, 3, 3)
        
        # Create a (2*in_channels, 1, 3, 3) weight tensor for grouped convolution
        # Since groups=in_channels, each group sees 1 input channel.
        final_weights = np.zeros((2 * in_channels, 1, 3, 3), dtype=np.float32)
        
        for i in range(in_channels):
            final_weights[i*2 : (i+1)*2, 0, :, :] = sobel_weights_xy

        # Create the Conv2d layer
        self.conv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        
        # Set the weights and make them non-trainable
        self.conv.weight = nn.Parameter(torch.from_numpy(final_weights), requires_grad=False)

    def forward(self, x):
        # Apply the fixed Sobel filter
        return self.conv(x)