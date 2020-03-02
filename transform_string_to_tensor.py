from typing import List
import torch
import numpy as np

from dictionary import Dictionary


class TransformStringToTensor:

    def __init__(self):
        self.dictionary = Dictionary.word_convert

    def transform_string_to_tensor(self, phrase: str) -> torch.float32:

        words = phrase.split()
        words_normalize = self.normalize(words)
        words_numbers = self.transform_word_to_int(words_normalize)
        words_numbers = self.adjustment_size(words_numbers)
        tensor = self.transform_to_tensor(words_numbers)

        return tensor

    def normalize(self, words: List[str]) -> List[str]:

        words_normalize = words

        return words_normalize

    def transform_word_to_int(self, words: List[str]) -> List[int]:

        numbers = []

        for word in words:
            if word in self.dictionary.keys():
                numbers.append(self.dictionary[word])
            else:
                numbers.append(1000)

        return numbers

    @staticmethod
    def adjustment_size(words_num: List[int]) -> List[int]:

        while len(words_num) < 50:
            words_num.append(0)

        return words_num

    @staticmethod
    def transform_to_tensor(words_num: List[int]) -> torch.float32:

        array = np.array(words_num)
        tensor = torch.from_numpy(array).float()

        return tensor
