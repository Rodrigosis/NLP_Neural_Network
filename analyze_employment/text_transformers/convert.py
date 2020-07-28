from numpy import array

from analyze_employment.text_transformers.characters import Characters


class Convert:

    def text_to_array(self, text: str) -> array:
        words = text.split()

        text_array = []
        for word in words:
            word_int = []
            for letter in word:
                word_int.append(Characters().from_character(letter))

            text_array.append(array(word_int))

        return array(text_array)

    def array_to_text(self, text_array: array) -> str:
        text = ''

        for word in text_array:
            for letter in word:
                character = Characters().from_number(letter)
                text = text + str(character)
            text = text + ' '

        return text
