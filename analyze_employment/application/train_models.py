from typing import List

from analyze_employment.tools.question_generator.neural_networks.nn_question_generator import QuestionGenerator
from analyze_employment.optimizers.question_generator.question_generator_optimizer import QuestionGeneratorOptimizer
from analyze_employment.text_transformers.convert import Convert


class TrainQuestionGenerator:

    def train(self, input_data: List[str], correct_output: List[str]):

        if len(input_data) != len(correct_output):
            raise ValueError('The input list must be the same size as the output list')

        input_data = Convert().text_to_array(input_data)
        correct_output = Convert().text_to_array(correct_output)

        QuestionGeneratorOptimizer().execute(network=QuestionGenerator(),
                                             data=input_data,
                                             correct_output=correct_output,
                                             batchs=2,
                                             epochs=10,
                                             learning_rate=1e6,
                                             weight_decay=0,
                                             file_name='network_QG')


if __name__ == '__main__':
    output_data = ["Do you have a bachelor’s Degree?", "Do you have a bachelor’s Degree?", "Do you have a bachelor’s Degree?", "Do you have a bachelor’s Degree?", "Do you have a bachelor’s Degree?", "Do you have a bachelor’s Degree?"]
    input_data = ["bachelor’s Degree", "bachelor’s Degree", "bachelor’s Degree", "bachelor’s Degree", "bachelor’s Degree", "bachelor’s Degree"]

    TrainQuestionGenerator().train(input_data, output_data)
