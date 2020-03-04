from typing import List
import torch

from nn_find_correct_piece import FindCorrectPiece
from nn_complete_requirement import CompleteRequirement
from nn_choose_type_question import ChooseTypeQuestion
from nn_confirms_quality import ConfirmsQuality
from optimizer import Optimizer
from transform_string_to_tensor import TransformStringToTensor


class ManagerNeuralNetworks:

    def __init__(self, find_correct_piece_parameters: str, complete_requirement_parameters: str,
                 choose_type_question_parameters: str, confirms_quality_parameters: str):
        self._transform = TransformStringToTensor()
        self._find_correct_piece = FindCorrectPiece()
        torch.save(self._find_correct_piece.state_dict(),
                   'neural_networks_parameters/' + find_correct_piece_parameters)
        self._complete_requirement = CompleteRequirement()
        torch.save(self._complete_requirement.state_dict(),
                   'neural_networks_parameters/' + complete_requirement_parameters)
        self._choose_type_question = ChooseTypeQuestion()
        torch.save(self._choose_type_question.state_dict(),
                   'neural_networks_parameters/' + choose_type_question_parameters)
        self._confirms_quality = ConfirmsQuality()
        torch.save(ConfirmsQuality().state_dict(),
                   'neural_networks_parameters/' + confirms_quality_parameters)

    def find_correct_piece(self, phrases: List[str]):
        tensor = self._transform.transform_string_to_tensor(phrases, False)
        forecast = self._find_correct_piece(tensor)
        return forecast

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
