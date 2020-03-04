if __name__ == '__main__':
    from manager_neural_networks import ManagerNeuralNetworks

    manager = ManagerNeuralNetworks('teste_save_epoch_9.pt', 'teste_save_epoch_9.pt',
                                    'teste_save_epoch_9.pt', 'teste_save_epoch_9.pt')

    frase = ['working knowledge of project and construction sequencing & scheduling',
             'working knowledge of project and construction sequencing & scheduling',
             'working knowledge of project and construction sequencing & scheduling',
             'working knowledge of project and construction sequencing & scheduling']

    resultado = manager.find_correct_piece(frase)
    print(resultado)

    correto = ['__ZERO__ knowledge of project and construction sequencing & scheduling',
               '__ZERO__ knowledge of project and construction sequencing & scheduling',
               '__ZERO__ knowledge of project and construction sequencing & scheduling',
               '__ZERO__ knowledge of project and construction sequencing & scheduling']

    from nn_find_correct_piece import FindCorrectPiece
    manager.optimizer(FindCorrectPiece(), frase, correto, 2, 10, 1e-6, 0, 'teste_save')

    testes = ['working knowledge of project and construction sequencing & scheduling',
              'the candidate should also possess knowledge of work performed in other labs and assist as needed',
              'candidate should possess experience with Windows 7, TCP/IP, and Web and Network LAN based systems',
              'proficient in microservices and building high performant, scalable applications',
              'individual must have a minimum of three years hotel, convention/conference or event '
              'planning experience required',
              'bachelor’s Degree in Finance or Accounting required (MBA preferred, not required)',
              'proficient in code versioning tools, such as GitHub',
              'bachelors Degree required',
              'minimum 5 years of experience in the physical security, fire alarm or low voltage '
              'communication industry installing field devices, programming systems, and running medium',
              'expert in HTML5, CSS3, JavaScript. Knowledge of third-party libraries like jQuery, '
              'NodeJS, Angular and React Js is must']
