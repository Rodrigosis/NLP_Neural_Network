from typing import List

from analyze_employment.text_transformers.characters import Characters


class Convert:

    def __init__(self):
        self.transform_text = Characters().from_character
        self.transform_array = Characters().from_number

    def text_to_array(self, text_list: List[str]) -> List[List]:
        data = []

        for text in text_list:
            text = text.replace('\/', '/')

            words = text.split()

            text_array = []
            for word in words:
                word_int = []
                for letter in word:
                    word_int.append(self.transform_text(letter))

                word_int = self._adjust_size(word_int, 50)

                text_array.append(word_int)

            while len(text_array) < 50:
                text_array.append(self._adjust_size([], 50))

            data.append(text_array)

        return data

    def array_to_text(self, text_array_list: List[List]) -> List[str]:
        data = []

        for text_array in text_array_list:
            text = ''

            for word in text_array:
                for letter in word:
                    text = text + str(self.transform_array(int(letter)))
                text = text + ' '

            data.append(text.strip())

        return data

    def _adjust_size(self, li: List, size: int) -> List:

        if len(li) > size:
            raise ValueError('The list is larger than the requested fit size')

        while len(li) < size:
            li.append(100)

        return li
