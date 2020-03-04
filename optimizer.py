from typing import List, Tuple, Dict
import random
from torch import optim

from transform_string_to_tensor import TransformStringToTensor


class Optimizer:

    def __init__(self):
        self.transform = TransformStringToTensor().transform_string_to_tensor

    def optimizer(self, network, data: List[str], correct_output: List[str], batchs: int, epochs: int,
                  learning_rate: float, weight_decay: float):

        data_tuples = self.create_tuple(data, correct_output)

        for epoch in range(epochs):
            batch_packages = self.create_batchs(data_tuples, batchs)

            for batch in batch_packages:

                pred_data = network(batch['data'])
                optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
                network_loss = network.criterion_l1_loss(pred_data, batch['correct_output'])
                network_loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(float(network_loss.data))

    def create_tuple(self, data: List[str], correct_output: List[str]) -> List[Tuple]:
        assert len(data) == len(correct_output)

        data_tuples = []

        for req, req_test in zip(data, correct_output):
            tensor_req = self.transform([req])
            tensor_req_test = self.transform([req_test])
            data_tuples.append((tensor_req, tensor_req_test))

        return data_tuples

    @staticmethod
    def create_batchs(data_tuples: List[Tuple], batch) -> List[Dict]:
        batchs_dict = []

        data_tuples = random.shuffle(data_tuples)
        batchs = [data_tuples[i::batch] for i in range(batch)]

        for package in batchs:
            data = []
            correct_output = []
            for i in package:
                data.append(i[0])
                correct_output.append(i[1])
            batchs_dict.append({'data': data, 'correct_output': correct_output})

        return batchs_dict
