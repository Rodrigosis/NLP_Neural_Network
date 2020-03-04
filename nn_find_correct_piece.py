import torch
from torch import nn


class FindCorrectPiece(nn.Module):
    def __init__(self):
        super(FindCorrectPiece, self).__init__()
        # torch.manual_seed(42)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input = nn.Linear(50, 200)
        self.relu = nn.ReLU()
        self.out = nn.Linear(200, 50)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.relu(self.input(x))
        output = self.sigmoid(self.out(feature))
        return output

    def criterion_l1_loss(self, predicted, real):
        criterion = nn.L1Loss().to(self.device)
        loss_l1 = criterion(predicted, real)
        return loss_l1

    def criterion_mse_loss(self, predicted, real):
        criterion = nn.MSELoss().to(self.device)
        loss_mse = criterion(predicted, real)
        return loss_mse
