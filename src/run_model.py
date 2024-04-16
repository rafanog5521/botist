#!/usr/bin/env python3
from model_interactor import TinyLlamaModelInteractor, PhiModelInteractor
from config.parameters import *
import json, argparse
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('--init_only', help='Trigger a sample run to download requirements', required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-q', '--questions', type=int, default=3, help='Amount of questions to evaluate the model(3 by default)', required=False)
args = vars(parser.parse_args())

parameters = PipelineParams()
if 'TinyLlama' in parameters.model or 'Phi' in parameters.model:
    if 'TinyLlama' in parameters.model:
        interactor = TinyLlamaModelInteractor()
    elif 'Phi' in parameters.model:
        interactor = PhiModelInteractor()
    
    if args['init_only']:
        interactor.init_model()
    else:
        with open(questions_path, 'r') as file:
            full_questionnaire = json.load(file)
            # Load a random subset of n questions to execute the tests
            question_numbers = [random.randint(0, len(full_questionnaire)) for _ in range(args["questions"])]
            questionnaire = []
            for n in question_numbers:
                questionnaire.append(full_questionnaire[n])

            progress_bar = tqdm(total=len(questionnaire), desc="Processing questionnaire...")

            for q in questionnaire:
                resp = interactor.ask_question(question=q)
                print ("QUESTION: ",q['content'])
                print ("ANSWER: ",resp['output'])
                print ("\n")
                q.update({"response": resp})
                progress_bar.update(1)
            progress_bar.close()

            print ("\nFULL QUESTIONNAIRE: ")
            print(questionnaire)
