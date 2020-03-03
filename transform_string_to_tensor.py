from typing import List
import torch
import numpy as np
import re
import json


class TransformStringToTensor:

    def __init__(self):
        with open('dictionary.json') as json_file:
            self.dictionary = json.load(json_file)

    def transform_string_to_tensor(self, phrases: List[str]) -> torch.float32:

        words = self.split_phrase(phrases)
        words_numbers = self.transform_word_to_int(words)
        tensor = self.transform_to_tensor(words_numbers)

        return tensor

    @staticmethod
    def split_phrase(phrases: List[str]) -> List[List[str]]:
        words = []

        for phrase in phrases:
            word = re.findall(r"[\w']+|[''.,ª&º@#|$°%!–`´<’>*?;/:“”(){}\+\-]", phrase)
            words.append(word)

        return words

    def transform_word_to_int(self, phrases: List[List[str]]) -> List:
        phrases_int = []

        for phrase in phrases:
            numbers = []
            for word in phrase:
                if word in self.dictionary.keys():
                    numbers.append(self.dictionary[word])
                else:
                    self.dictionary['numero_de_palavras'] += 1
                    self.dictionary[word] = self.dictionary['numero_de_palavras']
                    numbers.append(self.dictionary['numero_de_palavras'])

                    with open('dictionary.json', 'w') as outfile:
                        json.dump(self.dictionary, outfile)

            numbers = self.adjustment_size(numbers)
            phrases_int.append(np.array(numbers).astype(np.float32))

        return phrases_int

    @staticmethod
    def transform_to_tensor(words_num: List):

        array = np.array(words_num)
        tensor = torch.from_numpy(array).float()

        return tensor

    @staticmethod
    def adjustment_size(phrase: List[int]) -> List[int]:

        if len(phrase) > 50:
            phrase = []

        while len(phrase) < 50:
            phrase.append(0)

        return phrase
