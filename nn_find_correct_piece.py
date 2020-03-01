import torch
from torch import nn


class FindCorrectPiece(nn.Module):
    def __init__(self):
        super(FindCorrectPiece, self).__init__()
        torch.manual_seed(42)

        self.hidden = nn.Linear(50, 200)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(200, 50)
        self.hardtanh = nn.Hardtanh()

    def forward(self, x):
        feature = self.tanh(self.hidden(x))
        output = self.hardtanh(self.out(feature))

        return output

    def criterion(self):
        criterion = nn.CrossEntropyLoss()

        return


if __name__ == '__main__':
    from transform_string_to_tensor import TransformStringToTensor

    net = FindCorrectPiece()
    transform = TransformStringToTensor()

    frase = 'must have a valid driver’s license'

    tensor = transform.transform(frase)
    print(tensor)
    resultado = net.forward(tensor)

    print(resultado)
