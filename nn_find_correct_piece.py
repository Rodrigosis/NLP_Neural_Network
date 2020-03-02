import torch
from torch import nn
from torch import optim


class FindCorrectPiece(nn.Module):
    def __init__(self):
        super(FindCorrectPiece, self).__init__()
        torch.manual_seed(42)

        self.hidden = nn.Linear(50, 200)
        self.tanh = nn.ReLU()
        self.out = nn.Linear(200, 50)
        self.hardtanh = nn.Sigmoid()

    def forward(self, x):
        feature = self.tanh(self.hidden(x))
        output = self.hardtanh(self.out(feature))

        return output

    @staticmethod
    def criterion_l1_loss(predicted, real):

        criterion = nn.L1Loss()
        t = criterion(predicted, real)
        return t

    @staticmethod
    def criterion_mse_loss(predicted, real):

        criterion = nn.MSELoss()
        t = criterion(predicted, real)

        return t


def optimizer(rede, ):
    opt = optim.Adam(FindCorrectPiece().parameters(), lr=1e-3)

    return opt


if __name__ == '__main__':
    from transform_string_to_tensor import TransformStringToTensor

    net = FindCorrectPiece()
    transform = TransformStringToTensor()

    frase = 'must have a valid driver’s license'

    tensor = transform.transform_string_to_tensor(frase)
    print(tensor)

    resultado = net(tensor)
    print(resultado)

    desejado = [0, 0, 1, 1, 1, 1]
    desejado = transform.adjustment_size(desejado)
    desejado = transform.transform_to_tensor(desejado)
    print(desejado)

    print(resultado.shape)
    print(desejado.shape)

    loss = net.criterion_l1_loss(resultado[0], desejado[0])
    print(loss)

    loss2 = net.criterion_mse_loss(resultado[0], desejado[0])
    print(loss2)
