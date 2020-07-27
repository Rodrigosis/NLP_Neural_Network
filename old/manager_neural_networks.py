from typing import List
import torch

from old.nn_find_correct_piece import FindCorrectPiece
from old.nn_complete_requirement import CompleteRequirement
from old.nn_choose_type_question import ChooseTypeQuestion
from old.nn_confirms_quality import ConfirmsQuality
from old.optimizer import Optimizer
from old.transformers import TransformStringToTensor


class ManagerNeuralNetworks:

    def __init__(self):
        self._transform = TransformStringToTensor()
        self._find_correct_piece = FindCorrectPiece()
        self._complete_requirement = CompleteRequirement()
        self._choose_type_question = ChooseTypeQuestion()
        self._confirms_quality = ConfirmsQuality()

    def find_correct_piece(self, phrases: List[str]) -> List[str]:

        self._find_correct_piece.load_state_dict(
            torch.load('neural_networks_parameters/nn_find_correct_piece.pt'))

        tensor = self._transform.transform_string_to_tensor(phrases, False)
        forecast = self._find_correct_piece(tensor)

        output = []

        for phrase, vetor in zip(phrases, forecast):
            split = self._transform.split_phrase([phrase])
            out = ''
            for req, ten in zip(split[0], vetor):
                if ten > 0.5:
                    out = out + ' ' + req

            output.append(out)

        return output

    def teste(self, phrases: List[str], correct: List[str]):
        self._find_correct_piece.load_state_dict(
            torch.load('neural_networks_parameters/nn_find_correct_piece.pt'))

        tensor = self._transform.transform_string_to_tensor(phrases, False)
        tensor_correct = self._transform.transform_string_to_tensor(correct, True)
        forecast_1 = self._find_correct_piece.criterion_l1_loss(tensor, tensor_correct)
        forecast_2 = self._find_correct_piece.criterion_mse_loss(tensor, tensor_correct)

        return forecast_1, forecast_2

    def complete_requirement(self):
        pass

    def choose_type_question(self):
        pass

    def confirms_quality(self):
        pass

    @staticmethod
    def optimizer(network, data: List[str], correct_output: List[str], batchs: int, epochs: int,
                  learning_rate: float, weight_decay: float, file_name: str):
        Optimizer().optimizer(network, data, correct_output, batchs, epochs, learning_rate, weight_decay, file_name)
