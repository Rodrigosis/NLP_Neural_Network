if __name__ == '__main__':

    # epoch = 1000
    # batch = 10
    # lr = 1e-5
    # weight_decay=0
    # nn.Linear(50, 300)
    # nn.ReLU()
    # nn.Linear(300, 1000)
    # nn.ReLU()
    # nn.Linear(1000, 1000)
    # nn.ReLU()
    # nn.Linear(1000, 50)
    # nn.Sigmoid()

    from transform_string_to_tensor import TransformStringToTensor
    from nn_find_correct_piece import FindCorrectPiece
    from optimizer import Optimizer
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
