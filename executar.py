if __name__ == '__main__':
    from manager_neural_networks import ManagerNeuralNetworks
    import pandas as pd

    manager = ManagerNeuralNetworks()

    # req = ['basic understanding of revenue recognition and accounting a plus',
    #        'basic understanding of product compliance and global certifications',
    #        'basic understanding of the pharmaceutical industry',
    #        'basic SQL knowledge and database knowledge required',
    #        'basic supervisory and leadership skills. Proven successful interpersonal and customer skills required',
    #        'basic understanding of cabinetry layout, steel fabrication, and construction assembly techniques a plus',
    #        'basic computer skills/ basic excel skills',
    #        'basic knowledge of commercial real estate preferred']
    #
    # resultado = manager.find_correct_piece(req)
    #
    # for res in resultado:
    #     print(res)

    dados = pd.read_csv('C:/Users/nataly/Downloads/primeiro_teste.csv')

    requisitos = []
    corretos = []

    for req, corr in zip(dados['requirements'], dados['correto']):
        requisitos.append(req)
        corretos.append(corr)

    loss = manager.teste(requisitos, corretos)
    print(loss)

    from nn_find_correct_piece import FindCorrectPiece
    manager.optimizer(FindCorrectPiece(), requisitos, corretos, 5, 1000, 1e-6, 0, 'basic_questions')
