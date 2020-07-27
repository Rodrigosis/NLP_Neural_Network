import torch
from torch import nn


class FindCorrectPiece(nn.Module):
    def __init__(self):
        super(FindCorrectPiece, self).__init__()
        # with torch.no_grad():
        # torch.manual_seed(42)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input = nn.Linear(50, 1000)
        self.layer_one = nn.Linear(1000, 1000)
        self.layer_two = nn.Linear(1000, 1000)
        self.layer_three = nn.Linear(1000, 1000)
        self.layer_four = nn.Linear(1000, 1000)
        self.layer_five = nn.Linear(1000, 1000)
        self.layer_six = nn.Linear(1000, 1000)
        self.output = nn.Linear(1000, 50)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        step_1 = self.relu(self.input(x))
        step_2 = self.relu(self.layer_one(step_1))
        step_3 = self.relu(self.layer_two(step_2))
        step_4 = self.relu(self.layer_three(step_3))
        step_5 = self.relu(self.layer_four(step_4))
        step_6 = self.relu(self.layer_five(step_5))
        step_7 = self.relu(self.layer_six(step_6))
        output = self.sigmoid(self.output(step_7))
        return output

    def criterion_l1_loss(self, predicted, real):
        criterion = nn.L1Loss().to(self.device)
        loss_l1 = criterion(predicted, real)
        return loss_l1.data

    def criterion_mse_loss(self, predicted, real):
        criterion = nn.MSELoss().to(self.device)
        loss_mse = criterion(predicted, real)
        return loss_mse.data
