#!/usr/bin/env python3
from model_interactor import *
from config.parameters import *
from reports_module import Reporter
import json, argparse
from tqdm import tqdm
import random

if __name__ == "__main__":
    # Parameter section
    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('--init_only', help='Trigger a sample run to download requirements', required=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-b', '--base_test', help='Base test using the questions.json file specified', required=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-v', '--verbose', help='Verbose mode', required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-om', '--override_model', help='Override model from Param file', required=False, default=None, type=str)
    parser.add_argument('-od', '--override_dataset', help='Override dataset from Param file', required=False, default=None, type=str)
    args = vars(parser.parse_args())

    param = PipelineParams()
    #Override model
    if args["override_model"]:
        param.model = args["override_model"]
    #Override dataset
    if args["override_dataset"]:
        param.dataset = args["override_dataset"]

    # Check parameters
    print("==========\n")
    print("*\tUsing local model from {}".format(param.model)) if (param.local_model) else print("*\tUsing remote model from {}".format(param.model))
    print("\n*\tUsing local dataset from {}\n".format(param.dataset)) if (param.local_dataset) else print("\n*\tUsing remote dataset from {}\n".format(param.dataset)) 

    #####
    # Model instantiation
    if 'TinyLlama/TinyLlama' in param.model:
        interactor = TinyLlamaModelInteractor(pipeline_params=param)
        test_type = "ask_question"
    elif 'microsoft/phi-2' in param.model:
        interactor = PhiModelInteractor(pipeline_params=param)
        test_type = "ask_question"
    elif 'openai/whisper' in param.model:
        interactor = WhisperModelInteractor(pipeline_params=param)
        test_type = "transcription_of_speech" # "speech_evaluation" #transcription_of_speech
    else:
        test_type = None
        raise ValueError(f"{param.model} is not currently recognized as a model")
    
    #####
    # Condition to initiate model
    if args['init_only']:
        test_type = 'init_model'

    #####
    # Start to test
    match test_type:
        case 'init_model':
            interactor.init_model()

        case 'ask_question':
            # First we define the dataset to be used(either local or in cloud or administered by the dataset library)
            if args["base_test"]:
                with open(param.questions_path, 'r') as file:
                    full_questionnaire = json.load(file)
                    # Load a random subset of n questions to execute the tests
                    question_numbers = [random.randint(0, len(full_questionnaire)) for _ in range(param.num_prompts)]
                    questionnaire = []
                    for n in question_numbers:
                        questionnaire.append(full_questionnaire[n])
            else:  # aims to use a specific dataset using the dataset library
                if not hasattr(interactor, "dataset") or not hasattr(interactor, "dataset_subset"):
                    raise ValueError("No dataset or subset specified.")
                else:
                    data_interactor = DatasetInteractor(param, interactor.dataset, interactor.dataset_subset, interactor.dataset_split)
                    questionnaire = data_interactor.select_prompts_sample()  # to load the dataset to be used

            progress_bar = tqdm(total=len(questionnaire), desc="Processing prompts:")
            # Second we send the question to the model
            for q in questionnaire:
                resp = interactor.ask_question(question=q)
                q.update({"response": resp})
                progress_bar.update(1)
            progress_bar.close()

            # Third we generate the reports
            Reporter(param).process_results(questionnaire, True)

        case 'speech_evaluation':
            result, perf = interactor.evaluate_speech()
            print(perf)
            wer = perf("wer")
            print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

        case 'transcription_of_speech':
            data_interactor = DatasetInteractor(param, interactor.dataset, interactor.dataset_subset, interactor.dataset_split)
            transcription_array = data_interactor.select_prompts_sample()  # to load the dataset to be used
            progress_bar = tqdm(total=len(transcription_array), desc="Processing prompts:")
            # Second we send the question to the model
            for s in transcription_array:
                transcription = interactor.transcription_of_speech(speech=s)
                if args["verbose"]:
                    print("\n\n")
                    print("TRANSCRIPTION:")
                    print(transcription["current_response"])
                    print("\nEXPECTED RESPONSE:")
                    print(s["expected_response"])
                    print("\n")
                s.update({"response": transcription["current_response"]})
                s.update({"response_time": transcription["response_time"]})
                s.update({"tokens_per_second": transcription["tokens_per_sec"]})
                s.update({"WER": Reporter(param).calculate_wer_per_line(s["expected_response"], transcription["current_response"])})
                progress_bar.update(1)
            progress_bar.close()

            print(transcription_array)
            # Third we generate the reports
            transcription_array = [{key: value for key, value in dict.items() if (key != 'file' and key != 'audio')} for dict in transcription_array] #Remove file and audio paths to not break dict parsing functions
            Reporter(param).process_results(transcription_array, True, 'transcription')

        case None:
            print("What type of test do you want to run? Check configuration files")
            assert False
