#!/usr/bin/env python3
from model_interactor import TinyLlamaModelInteractor
from config import parameters
import json, argparse

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('--init_only', help='Trigger a sample run to download requirements', required=False, action=argparse.BooleanOptionalAction)
args = vars(parser.parse_args())

if 'TinyLlama' in parameters.model:
    interactor = TinyLlamaModelInteractor()
    if args['init_only']:
        interactor.init_model()
    else:
        question = open(parameters.questions_path, 'r')
        
        with open(parameters.questions_path, 'r') as file:
            questionnaire = json.load(file)
            for q in questionnaire:
                print("\n")
                print(interactor.ask_question(question=q))
