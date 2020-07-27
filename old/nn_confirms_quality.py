import torch
from torch import nn


class ConfirmsQuality(nn.Module):
    def __init__(self):
        super(ConfirmsQuality, self).__init__()
        # torch.manual_seed(42)

    def forward(self, x):
        pass

    def criterion_l1_loss(self):
        pass

    def criterion_mse_loss(self):
        pass
