from config import parameters
from datasets import load_dataset
import transformers
from transformers import pipeline
from tqdm import tqdm
import time



class TinyLlamaModelInteractor:
    def __init__(self, answers=parameters.num_return_sequences):
        self.__pipe__ = pipeline(task=parameters.task, model=parameters.model,
                                 torch_dtype=parameters.torch_dtype, device_map=parameters.device_map,
                                 num_return_sequences=answers)
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)  # disable base warnings

    def prompt(self, question):
        new_q = [question]
        return self.__pipe__.tokenizer.apply_chat_template(new_q, tokenize=parameters.tokenize,
                                                           add_generation_prompt=parameters.add_generation_prompt)

    def init_model(self, question='Say Hello'): #Use a sample question to trigger downloads for Tinyllama resources.
        self.prompt(question)

    def ask_question(self, question, performance_metric=True):
        prompt = self.prompt(question)
        if performance_metric:
            start_time = time.time()

        output =  self.__pipe__(prompt, max_new_tokens=parameters.max_new_tokens, do_sample=parameters.do_sample,
                                temperature=parameters.temperature, top_k=parameters.top_k, top_p=parameters.top_p)
        response = {"output": output[0]["generated_text"].split('<|assistant|>\n')[1]}

        if performance_metric:
            end_time = time.time()
            # Calculate response time
            response_time = end_time - start_time
            response.update({"response_time": response_time})
            # Calculate tokens per second
            total_tokens_generated = len(output[0]["generated_text"])
            tokens_per_second = total_tokens_generated / response_time
            response.update({"tokens_per_second": tokens_per_second})

        return response

class DatasetInteractor:
    def __init__(self):
        try:
            print(f"Loading \"{parameters.datasets_path}\" as dataset to be used")
            self.dataset = load_dataset(parameters.datasets_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        else:
            self.dataset_subset = parameters.dataset_subset  # this select a particular subset(MIGHT BE SELECTED RANDOMLY)
            self.dataset = self.dataset[self.dataset_subset]

    def process_dataset_format(self, data):  # This is to standardize the format of the prompt list for report purpose
        progress_bar = tqdm(total=len(data), desc="Formatting dataset:")
        if "ultrafeedback_binarized" in parameters.datasets_path:
            processed_data = []
            for p in data:
                prompt = {"role": "user", "content": p["prompt"], "prompt_id": p["prompt_id"],
                          "expected_response": p["chosen"][1]}
                processed_data.append(prompt)
                progress_bar.update(1)
            progress_bar.close()
            return processed_data
        else:
            print(f"{parameters.datasets_path} is currently not recognized by the framework...")
            raise
    def select_prompts_sample(self):
        # We filter the dataset to narrow the amount of prompts(selecting scores accordingly to
        # what is defined in the parameters)
        print(f"Selecting randomized samples from \"{parameters.dataset_subset}\" subset")
        filtered_dataset = self.dataset.filter(lambda example: example["score_chosen"] >= parameters.score_base)
        filtered_dataset = filtered_dataset.shuffle()  #shuffled to randomize it
        random_sample = filtered_dataset.select(range(parameters.num_prompts))
        return self.process_dataset_format(random_sample)
