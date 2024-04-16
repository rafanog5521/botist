#!/usr/bin/env python3
from model_interactor import TinyLlamaModelInteractor
from model_interactor import DatasetInteractor
from config import parameters
import json, argparse
from tqdm import tqdm
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('--init_only', help='Trigger a sample run to download requirements', required=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-b', '--base_test', help='Base test using the questions.json file specified', required=False,
                        action=argparse.BooleanOptionalAction)
    args = vars(parser.parse_args())

    if 'TinyLlama' in parameters.model:
        interactor = TinyLlamaModelInteractor()
    else:
        raise ValueError(f"{parameters.model} is not currently recognized as a model")

    if args['init_only']:
        interactor.init_model()
    else:
        # First: we load the dataset, either from a local file or one accesible through the datasets module
        if not args["base_test"]:
            data_interactor = DatasetInteractor()
            questionnaire = data_interactor.select_prompts_sample()  # to load the dataset to be used
        else:
            with open(parameters.questions_path, 'r') as file:
                full_questionnaire = json.load(file)
                # Load a random subset of n questions to execute the tests
                question_numbers = [random.randint(0, len(full_questionnaire)) for _ in range(parameters.num_prompts)]
                questionnaire = []
                for n in question_numbers:
                    questionnaire.append(full_questionnaire[n])

        progress_bar = tqdm(total=len(questionnaire), desc="Processing prompts:")

        # Second: We send the model the questions
        for q in questionnaire:
            resp = interactor.ask_question(question=q)
            q.update({"response": resp})
            progress_bar.update(1)
        progress_bar.close()

        # Third: We process the results
        for r in questionnaire:
            print(r)

