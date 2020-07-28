from typing import List, Tuple, Dict
import random
import torch
from torch import optim


class Optimizer:

    def execute(self, network, data: List[str], correct_output: List[str], batchs: int, epochs: int,
                learning_rate: float, weight_decay: float, file_name: str):

        best_epoch = 1e6
        data_tuples = self.create_tuple(data, correct_output)

        for epoch in range(epochs):
            batch_packages = self.create_batchs(data_tuples, batchs)
            loss = 0
            for batch in batch_packages:
                pred_data = network(batch['data'])
                optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
                network_loss = loss = network.criterion_l1_loss(pred_data, batch['correct_output'])
                network_loss.backward()
                optimizer.step()

            print(f'{float(loss.data)} epoch: {epoch}')

            if float(loss.data) <= best_epoch:
                best_epoch = float(loss.data)
                torch.save(network.state_dict(), 'optimizer_train/' + file_name + '_epoch_' + str(epoch) + '.pt')

        print(f'Melhor resultado: {best_epoch}')

    def create_tuple(self, data: List[str], correct_output: List[str]) -> List[Tuple]:
        assert len(data) == len(correct_output)

        data_tuples = []

        return data_tuples

    @staticmethod
    def create_batchs(data_tuples: List[Tuple], batch: int) -> List[Dict]:
        batchs_dict = []

        random.shuffle(data_tuples)
        batchs = [data_tuples[i::batch] for i in range(batch)]

        for package in batchs:
            data = []
            correct_output = []
            for i in package:
                data.append(i[0])
                correct_output.append(i[1])

            batchs_dict.append({'data': data[0], 'correct_output': correct_output[0]})

        return batchs_dict
