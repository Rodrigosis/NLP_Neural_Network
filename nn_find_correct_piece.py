import torch
from torch import nn
from torch import optim


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


def optimizer(net, x, real_x):
    device = torch.device('cpu')
    x.to(device)
    real_x.to(device)
    net.to(device)

    for i in range(1000):
        pred_x = net(x).to(device)
        opt = optim.Adam(FindCorrectPiece().parameters(), lr=1e-3, weight_decay=0)
        net_loss = net.criterion_l1_loss(pred_x, real_x)
        net_loss.backward()
        opt.step()

        if i % 100 == 0:
            print(float(net_loss.data))


if __name__ == '__main__':
    from transform_string_to_tensor import TransformStringToTensor
    import pandas as pd
    import numpy as np

    dados = pd.read_csv('C:/Users/nataly/OneDrive/Documentos/requirement_vetor_binario.csv', index_col=False)

    net = FindCorrectPiece()
    transform = TransformStringToTensor()

    frases = []
    for frase in dados['texto']:
        frases.append(frase)

    binarios = []
    for lista in dados['binario']:
        li = []
        for i in lista:
            if i == '1':
                li.append(1)
            elif i == '0':
                li.append(0)
        li = transform.adjustment_size(li)
        binarios.append(np.array(li).astype(np.float32))

    tensor = transform.transform_string_to_tensor(frases)
    resultado = net(tensor)

    desejado = transform.transform_to_tensor(binarios)

    print(resultado.shape)
    print(desejado.shape)

    loss = net.criterion_l1_loss(resultado, desejado)
    print(loss)

    loss2 = net.criterion_mse_loss(resultado, desejado)
    print(loss2)

    optimizer(net, tensor, desejado)
