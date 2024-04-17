from datasets import load_dataset
from config.parameters import *
import transformers
import torch, time
from tqdm import tqdm
#Tinyllama
from transformers import pipeline
#Phi
from transformers import AutoModelForCausalLM, AutoTokenizer

class TinyLlamaModelInteractor:
    def __init__(self):
        self.tiny_param = TinyLlamaParameters()
        self.pipe_param = PipelineParams()
        self.__pipe__ = pipeline(task=self.pipe_param.task, model=self.pipe_param.model,
                                 torch_dtype=self.pipe_param.torch_dtype, device_map=self.pipe_param.device_map,
                                 num_return_sequences=self.tiny_param.num_return_sequences)
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)  # disable base warnings
        # Might need to include a parameter while initiating the interactor class to differentiate when used in init_model mode or isolate this into a different method
        # self.data_interactor = DatasetInteractor(self.tiny_param.dataset, self.tiny_param.dataset_subset).select_prompts_sample()

    def prompt(self, question):
        new_q = [question]
        return self.__pipe__.tokenizer.apply_chat_template(new_q, tokenize=self.tiny_param.tokenize,
                                                           add_generation_prompt=self.tiny_param.add_generation_prompt)

    def init_model(self, question='Say Hello'): #Use a sample question to trigger downloads for Tinyllama resources.
        self.prompt(question)

    def ask_question(self, question, performance_metric=True):
        prompt = self.prompt(question)
        if performance_metric:
            start_time = time.time()

        output =  self.__pipe__(prompt, max_new_tokens=self.tiny_param.max_new_tokens, do_sample=self.tiny_param.do_sample,
                                temperature=self.tiny_param.temperature, top_k=self.tiny_param.top_k, top_p=self.tiny_param.top_p)
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

class PhiModelInteractor:
    def __init__(self):
        self.pipe_param = PipelineParams()
        self.phi_param = PhiParameters()
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=self.pipe_param.torch_dtype, trust_remote_code=self.phi_param.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=self.phi_param.trust_remote_code)
        #torch.set_default_device("cuda")

    def prompt(self, question):
        return self.tokenizer(question, return_tensors="pt", return_attention_mask=self.phi_param.return_attention_mask)

    def init_model(self, question='Say Hello'): #Use a sample question to trigger downloads for Phi resources.
        self.prompt(question)

    def ask_question(self, question, performance_metric=True):
        prompt = self.prompt("Instruct: "+question['content']+"\nOutput")
        if performance_metric:
            start_time = time.time()
        output = self.model.generate(**prompt, max_length=self.phi_param.max_length)
        readable_output = (self.tokenizer.batch_decode(output)[0].split('\n<|endoftext|>'))[0]
        readable_output = readable_output.split("\nOutput: ")[1]
        response = {"output": readable_output}

        if performance_metric:
            end_time = time.time()
            # Calculate response time
            response_time = end_time - start_time
            response.update({"response_time": response_time})
            # Calculate tokens per second
            total_tokens_generated = len(output)
            tokens_per_second = total_tokens_generated / response_time
            response.update({"tokens_per_second": tokens_per_second})

        return response

class DatasetInteractor:
    def __init__(self, path, subset):
        try:
            print(f"Loading \"{path}\" as dataset to be used")
            self.dataset = load_dataset(path)
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

