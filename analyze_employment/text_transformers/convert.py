from typing import List
from numpy import array

from analyze_employment.text_transformers.characters import Characters


class Convert:

    def __init__(self):
        self.transform_text = Characters().from_character
        self.transform_array = Characters().from_number

    def text_to_array(self, text_list: List[str]) -> List[array]:
        data = []

        for text in text_list:
            words = text.split()

            text_array = []
            for word in words:
                word_int = []
                for letter in word:
                    word_int.append(self.transform_text(letter))

                text_array.append(array(word_int))

            data.append(array(text_array))

        return data

    def array_to_text(self, text_array_list: List[array]) -> List[str]:
        data = []

        for text_array in text_array_list:
            text = ''

            for word in text_array:
                for letter in word:
                    text = text + str(self.transform_array(letter))
                text = text + ' '

            data.append(text)

        return data
