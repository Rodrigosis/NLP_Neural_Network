from typing import List
import torch
import numpy as np
import re
import json


class TransformStringToTensor:

    def __init__(self):
        with open('dictionary.json') as json_file:
            self.dictionary = json.load(json_file)

    def transform_string_to_tensor(self, phrase: str) -> torch.float32:

        words = self.split_phrase(phrase)
        words_numbers = self.transform_word_to_int(words)
        words_numbers = self.adjustment_size(words_numbers)
        tensor = self.transform_to_tensor(words_numbers)

        return tensor

    @staticmethod
    def split_phrase(phrase: str) -> List[str]:

        words = re.findall(r"[\w']+|[.,!?;/:(){}]", phrase)

        return words

    def transform_word_to_int(self, words: List[str]) -> List[int]:

        numbers = []

        for word in words:
            if word in self.dictionary.keys():
                numbers.append(self.dictionary[word])
            else:
                self.dictionary['numero_de_palavras'] += 1
                self.dictionary[word] = self.dictionary['numero_de_palavras']
                numbers.append(self.dictionary['numero_de_palavras'])

                with open('dictionary.json', 'w') as outfile:
                    json.dump(self.dictionary, outfile)

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
