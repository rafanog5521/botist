#!/usr/bin/env python3
from model_interactor import TinyLlamaModelInteractor
from config import parameters
import json
from tqdm import tqdm
import random

question_amount = 3

if 'TinyLlama' in parameters.model:
    interactor = TinyLlamaModelInteractor()

    question = open(parameters.questions_path, 'r')

    with open(parameters.questions_path, 'r') as file:
        full_questionnaire = json.load(file)
        # Load a random subset of n questions to execute the tests
        question_numbers = [random.randint(0, len(full_questionnaire)) for _ in range(question_amount)]
        questionnaire = []
        for n in question_numbers:
            questionnaire.append(full_questionnaire[n])

        progress_bar = tqdm(total=len(questionnaire), desc="Processing questionnaire...")

        for q in questionnaire:
            resp = interactor.ask_question(question=q)
            q.update({"response": resp})
            progress_bar.update(1)
        progress_bar.close()