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


if __name__ == '__main__':
    from transform_string_to_tensor import TransformStringToTensor
    import pandas as pd
    import numpy as np

    dados = pd.read_csv('C:/Users/nataly/OneDrive/Documentos/requirement_vetor_binario.csv', index_col=False)

    net = FindCorrectPiece()
    transform = TransformStringToTensor()

    frases = []
    for frase in dados['texto']:
        frases.append(str(frase))

    lista = []
    for frase in frases:
        lenn = transform.split_phrase([frase])
        if len(lenn[0]) <= 45:
            lista.append(frase)

    df = pd.DataFrame(lista)
    #df.to_csv('C:/Users/nataly/OneDrive/Documentos/dataset_de_treino_len_45.csv', header=False)

    print(net.parameters())
    print(net)


    # tensor = transform.transform_string_to_tensor(frases)
    # resultado = net(tensor)

    # print(resultado.shape)

    # loss = net.criterion_l1_loss(resultado, desejado)
    # print(loss)
    #
    # loss2 = net.criterion_mse_loss(resultado, desejado)
    # print(loss2)
    #
    # optimizer(net, tensor, desejado)
    #
    # testes = ['working knowledge of project and construction sequencing & scheduling',
    #           'the candidate should also possess knowledge of work performed in other labs and assist as needed',
    #           'candidate should possess experience with Windows 7, TCP/IP, and Web and Network LAN based systems',
    #           'proficient in microservices and building high performant, scalable applications',
    #           'individual must have a minimum of three years hotel, convention/conference or event planning experience required',
    #           'bachelor’s Degree in Finance or Accounting required (MBA preferred, not required)',
    #           'proficient in code versioning tools, such as GitHub',
    #           'bachelors Degree required',
    #           'minimum 5 years of experience in the physical security, fire alarm or low voltage communication industry installing field devices, programming systems, and running medium',
    #           'expert in HTML5, CSS3, JavaScript. Knowledge of third-party libraries like jQuery, NodeJS, Angular and React Js is must']
    #
    # for i in testes:
    #
    #     tensor = transform.transform_string_to_tensor([i])
    #     res = net(tensor)
    #
    #     req = transform.split_phrase([i])
    #
    #     frase = ''
    #     for resp, req in zip(res[0], req[0]):
    #         if resp >= 0.5:
    #             frase = frase + ' ' + req
    #
    #     print(i)
    #     print(frase.strip())
    #     print(' ')
