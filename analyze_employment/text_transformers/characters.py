class Characters:

    def __init__(self):
        self.character = {
            '': 0,
            'a': 1,
            'A': 2,
            'b': 3,
            'B': 4,
            'c': 5,
            'C': 6,
            'd': 7,
            'D': 8,
            'e': 9,
            'E': 10,
            'f': 11,
            'F': 12,
            'g': 13,
            'G': 14,
            'h': 15,
            'H': 16,
            'i': 17,
            'I': 18,
            'j': 19,
            'J': 20,
            'k': 21,
            'K': 22,
            'l': 23,
            'L': 24,
            'm': 25,
            'M': 26,
            'n': 27,
            'N': 28,
            'o': 29,
            'O': 30,
            'p': 31,
            'P': 32,
            'q': 33,
            'Q': 34,
            'r': 35,
            'R': 36,
            's': 37,
            'S': 38,
            't': 39,
            'T': 40,
            'u': 41,
            'U': 42,
            'v': 43,
            'V': 44,
            'w': 45,
            'W': 46,
            'x': 47,
            'X': 48,
            'y': 49,
            'Y': 50,
            'z': 51,
            'Z': 52,
            '!': 53,
            '@': 54,
            '#': 55,
            '$': 56,
            '%': 57,
            '&': 58,
            '*': 59,
            '(': 60,
            ')': 61,
            '-': 62,
            '_': 63,
            '=': 64,
            '+': 65,
            '"': 66,
            "'": 67,
            '|': 68,
            '/': 69,
            '’': 70,
            ',': 71,
            '.': 72,
            '<': 73,
            '>': 74,
            ':': 75,
            ';': 76,
            '?': 77,
            '[': 78,
            ']': 79,
            '{': 80,
            '}': 81,
            '°': 82,
            'ª': 83,
            '´': 84,
            '1': 85,
            '2': 86,
            '3': 87,
            '4': 88,
            '5': 89,
            '6': 90,
            '7': 91,
            '8': 92,
            '9': 93,
            '0': 94
        }

    def from_character(self, letter: str) -> int:
        try:
            return self.character[letter]
        except:
            raise ValueError(f'Characters "{letter}" is unknown')

    def from_number(self, letter: int) -> str:
        char = ''

        for item in self.character.items():
            if item[1] == letter:
                char = item[0]
                break

        if char or letter == 0:
            return char
        else:
            raise ValueError(f'Characters "{letter}" is unknown')
