from typing import List
import torch
import numpy as np
import re
import json


class TransformStringToTensor:

    def __init__(self):
        with open('dictionary.json') as json_file:
            self.dictionary = json.load(json_file)

    def transform_string_to_tensor(self, phrases: List[str], output_data: bool) -> torch.float32:

        words = self.split_phrase(phrases)
        words_numbers = self.transform_word_to_int(words, output_data)
        tensor = self.transform_to_tensor(words_numbers)

        return tensor

    @staticmethod
    def split_phrase(phrases: List[str]) -> List[List[str]]:
        words = []

        for phrase in phrases:
            word = re.findall(r"[\w']+|[''.,ª&º@#|$°%!–`´<’>*?;/:“”(){}\+\-]", phrase)
            words.append(word)

        return words

    def transform_word_to_int(self, phrases: List[List[str]], output_data: bool) -> List:
        phrases_int = []

        for phrase in phrases:
            numbers = []
            for word in phrase:
                if word in self.dictionary.keys():
                    numbers.append(self.dictionary[word])
                else:
                    self.dictionary['__WORDS__'] += 1
                    self.dictionary[word] = self.dictionary['__WORDS__']
                    numbers.append(self.dictionary['__WORDS__'])

                    with open('dictionary.json', 'w') as outfile:
                        json.dump(self.dictionary, outfile)

            numbers = self.adjustment_size(numbers, output_data)
            phrases_int.append(np.array(numbers).astype(np.float32))

        return phrases_int

    @staticmethod
    def transform_to_tensor(words_num: List):

        array = np.array(words_num)
        tensor = torch.from_numpy(array).float()

        return tensor

    @staticmethod
    def adjustment_size(phrase: List[int], output_data: bool) -> List[int]:

        assert len(phrase) <= 50

        if output_data:
            phrase_output_data = []
            for i in phrase:
                if i == 0:
                    phrase_output_data.append(0)
                else:
                    phrase_output_data.append(1)

            phrase = phrase_output_data

        while len(phrase) < 50:
            phrase.append(0)

        return phrase

    def standardize_string_size(self, original: str, correct: str) -> str:

        assert correct in original

        output = correct

        split_original = self.split_phrase([original])
        split_correct = self.split_phrase([correct])

        n = len(split_correct)

        for i in range(len(split_original)):
            if split_correct[:n] == split_original[i:n+i]:
                break
            else:
                output = '__ZERO__ ' + output

        return output
