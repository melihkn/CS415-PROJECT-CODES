import torch
import torch.nn as nn

class DifferenceAttentionModule(nn.Module):
    """
    The 'Difference attention' block from the diagram.
    It learns a mask to apply to the absolute difference.
    """
    def __init__(self, in_channels):
        super(DifferenceAttentionModule, self).__init__()
        # A simple gate to learn the attention mask
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f_t1, f_t2):
        f_concat = torch.cat([f_t1, f_t2], dim=1)
        f_diff = torch.abs(f_t1 - f_t2)
        
        attention_mask = self.attention_gate(f_concat)
        
        # Refine the difference by multiplying it with the learned mask
        return f_diff * attention_mask