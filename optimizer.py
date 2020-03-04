from typing import List, Tuple, Dict
import random
import torch
from torch import optim

from transform_string_to_tensor import TransformStringToTensor


class Optimizer:

    def __init__(self):
        self.transform = TransformStringToTensor().transform_string_to_tensor

    def optimizer(self, network, data: List[str], correct_output: List[str], batchs: int, epochs: int,
                  learning_rate: float, weight_decay: float, file_name: str):

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

            print(float(loss.data))

            torch.save(network.state_dict(), 'optimizer_train/' + file_name + '_epoch_' + str(epoch) + '.pt')

    def create_tuple(self, data: List[str], correct_output: List[str]) -> List[Tuple]:
        assert len(data) == len(correct_output)

        data_tuples = []

        for req, req_test in zip(data, correct_output):
            tensor_req = self.transform([req], False)
            tensor_req_test = self.transform([req_test], True)
            data_tuples.append((tensor_req, tensor_req_test))

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
