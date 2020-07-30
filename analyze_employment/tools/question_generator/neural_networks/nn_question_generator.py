import torch
from torch import nn


class QuestionGenerator(nn.Module):
    def __init__(self):
        super(QuestionGenerator, self).__init__()
        with torch.no_grad():
            torch.manual_seed(42)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input = nn.Embedding(2500, 100) #2500, 100
        self.layer_one = nn.Linear(100, 1) #100, 50
        self.relu = nn.ReLU()

        # self.input = nn.Linear(50, 250)
        # self.layer_one = nn.Linear(250, 250)
        # self.output = nn.Linear(250, 50)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = torch.LongTensor(x)
        output = self.relu(self.input(input))
        output = self.relu(self.layer_one(output))

        # inout = self.relu(self.input(x))
        # step_1 = self.relu(self.layer_one(inout))
        # output = self.sigmoid(self.output(step_1))
        return output

    def criterion_l1_loss(self, predicted, real):
        r = []
        for i in real:
            x = []
            for ii in i:
                xx = []
                for iii in ii:
                    xx.append([iii])
                x.append(xx)
            r.append(x)

        real = torch.LongTensor(r)
        criterion = nn.L1Loss().to(self.device)
        loss_l1 = criterion(predicted, real)
        return loss_l1.data

    def criterion_mse_loss(self, predicted, real):
        criterion = nn.MSELoss().to(self.device)
        loss_mse = criterion(predicted, real)
        return loss_mse.data
