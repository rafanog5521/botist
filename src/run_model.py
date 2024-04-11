#!/usr/bin/env python3
from model_interactor import TinyLlamaModelInteractor
from config import parameters
import json

if 'TinyLlama' in parameters.model:
    interactor = TinyLlamaModelInteractor()

    question = open(parameters.questions_path, 'r')
    
    with open(parameters.questions_path, 'r') as file:
        questionnaire = json.load(file)
        for q in questionnaire:
            print("\n")
            print(interactor.ask_question(question=q))
