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

        self.input = nn.Linear(50, 300)
        self.layer_one = nn.Linear(300, 1000)
        self.layer_two = nn.Linear(1000, 1000)
        self.output = nn.Linear(1000, 50)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        step_1 = self.relu(self.input(x))
        step_2 = self.relu(self.layer_one(step_1))
        step_3 = self.relu(self.layer_two(step_2))
        output = self.sigmoid(self.output(step_3))
        return output

    def criterion_l1_loss(self, predicted, real):
        criterion = nn.L1Loss().to(self.device)
        loss_l1 = criterion(predicted, real)
        return loss_l1

    def criterion_mse_loss(self, predicted, real):
        criterion = nn.MSELoss().to(self.device)
        loss_mse = criterion(predicted, real)
        return loss_mse
