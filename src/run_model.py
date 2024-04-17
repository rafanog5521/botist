#!/usr/bin/env python3
from model_interactor import *
from config.parameters import *
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
    
    parameters = PipelineParams()
    if 'TinyLlama' in parameters.model:
        interactor = TinyLlamaModelInteractor()
    elif 'Phi' in parameters.model:
        interactor = PhiModelInteractor()
    else:
        raise ValueError(f"{parameters.model} is not currently recognized as a model")
    
    if args['init_only']:
        interactor.init_model()
    else:
        # First we define the dataset to be used(either local or in cloud or administered by the dataset library)
        if args["base_test"]:
            with open(questions_path, 'r') as file:
                full_questionnaire = json.load(file)
                # Load a random subset of n questions to execute the tests
                question_numbers = [random.randint(0, len(full_questionnaire)) for _ in range(num_prompts)]
                questionnaire = []
                for n in question_numbers:
                    questionnaire.append(full_questionnaire[n])
        else:  # aims to use a specific dataset using the dataset library
            if not hasattr(interactor, "dataset") or not hasattr(interactor, "dataset_subset"):
                raise ValueError("No dataset or subset specified.") 
            else:
                data_interactor = DatasetInteractor(interactor.dataset, interactor.dataset_subset)
                questionnaire = data_interactor.select_prompts_sample()  # to load the dataset to be used      

        progress_bar = tqdm(total=len(questionnaire), desc="Processing prompts:")
        # Second we send the question to the model
        for q in questionnaire:
            resp = interactor.ask_question(question=q)
            print ("QUESTION: ", q['content'])
            print ("ANSWER: ", resp['output'])
            print ("\n")
            q.update({"response": resp})
            progress_bar.update(1)
        progress_bar.close()

        # Third we generate the reports
        print ("\nFULL QUESTIONNAIRE: ")
        print(questionnaire)
