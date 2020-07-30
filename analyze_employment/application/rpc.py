from typing import List

from analyze_employment.tools.question_generator.neural_networks.nn_question_generator import QuestionGenerator
from analyze_employment.text_transformers.convert import Convert


class Build:

    def question(self, phrases: List[str]):
        phrases = Convert().text_to_array(phrases)
        result = QuestionGenerator().forward(phrases)

        return Convert().array_to_text(result)


if __name__ == '__main__':
    input_data = ["Bachelor’s Degree", "Bachelor’s Degree"]

    result = Build().question(input_data)
    print(result)
