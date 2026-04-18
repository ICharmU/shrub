import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, apply_sigmoid=True):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits) if self.apply_sigmoid else logits
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 2 * intersection / (union + self.smooth)
        return 1 - dice