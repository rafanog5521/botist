from datasets import load_dataset
from config.parameters import *
import transformers
import time
import re
from tqdm import tqdm
#Tinyllama
from transformers import pipeline
#Phi
from transformers import AutoModelForCausalLM, AutoTokenizer
#Whisper
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load

pipe_param = PipelineParams()

class TinyLlamaModelInteractor:
    def __init__(self):
        self.tiny_param = TinyLlamaParameters()
        self.__pipe__ = pipeline(task=pipe_param.task, model=pipe_param.model,
                                 torch_dtype=pipe_param.torch_dtype, device_map=pipe_param.device_map,
                                 num_return_sequences=self.tiny_param.num_return_sequences)
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)  # disable base warnings
        self.dataset = self.tiny_param.dataset
        self.dataset_subset = self.tiny_param.dataset_subset
        torch.set_default_device("cuda")

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
        self.phi_param = PhiParameters()
        self.model = AutoModelForCausalLM.from_pretrained(pipe_param.model, torch_dtype=pipe_param.torch_dtype, trust_remote_code=self.phi_param.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(pipe_param.model, trust_remote_code=self.phi_param.trust_remote_code)
        self.dataset = self.phi_param.dataset
        self.dataset_subset = self.phi_param.dataset_subset
        # self.device = torch.device("cuda:0")
        # self.model.cuda()
        # torch.set_default_device("cuda")

    def prompt(self, question):
        return self.tokenizer(question, return_tensors="pt", return_attention_mask=self.phi_param.return_attention_mask)
        #return self.tokenizer(question, return_tensors="pt", return_attention_mask=self.phi_param.return_attention_mask).to('cuda')

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

class WhisperModelInteractor:
    def __init__(self):
        self.whisper_param = WhisperParameters()
        self.dataset = self.whisper_param.dataset
        self.dataset_subset = self.whisper_param.dataset_subset
        self.dataset_split = self.whisper_param.dataset_split
        self.dataset_loaded = load_dataset(self.whisper_param.dataset, self.dataset_subset, split=self.dataset_split, trust_remote_code=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(pipe_param.model).to("cuda")
        self.processor = WhisperProcessor.from_pretrained(pipe_param.model)

    def map_to_pred(self, batch):
        audio = batch["audio"]
        input_features = self.processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        batch["reference"] = self.processor.tokenizer._normalize(batch['text'])

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features.to("cuda"))[0]
        transcription = self.processor.decode(predicted_ids)
        batch["prediction"] = self.processor.tokenizer._normalize(transcription)
        return batch

    def init_model(self):
        pass

    def evaluate_speech(self):
        result = self.dataset_loaded.map(self.map_to_pred)
        return (result, load)

    def transcription_of_speech(self, speech):
        # load dummy dataset and read audio files
        sample = speech["audio"]
        input_features = self.processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

        # generate token ids
        predicted_ids = self.model.generate(input_features.to("cuda"))[0]
        # decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        #Format output
        transcription.remove('<|startoftranscript|>')
        transcription.remove('<|notimestamps|>')
        transcription.remove('<|endoftext|>')
        transcription[0] = transcription[0].strip()
        readable_transcription = ','.join(map(str, transcription))
        readable_transcription = (re.sub(",", "", readable_transcription)).upper()
        return (readable_transcription, speech["text"])

class DatasetInteractor:
    def __init__(self, dataset, subset):
        self.dataset = load_dataset(dataset, subset, split='validation')
        self.dataset_subset = subset  # this select a particular subset(MIGHT BE SELECTED RANDOMLY)

    def process_dataset_format(self, data):  # This is to standardize the format of the prompt list for report purpose
        progress_bar = tqdm(total=len(data), desc="Formatting dataset:")
        if "ultrafeedback_binarized" in pipe_param.dataset_name:
            processed_data = []
            for p in data:
                prompt = {"role": "user", "content": p["prompt"], "prompt_id": p["prompt_id"],
                          "expected_response": p["chosen"][1]}
                processed_data.append(prompt)
                progress_bar.update(1)
            progress_bar.close()
            return processed_data
        elif "librispeech" in pipe_param.dataset_name:
            processed_data = []
            for p in data:
                prompt = {"file": p["file"], "audio": p["audio"],
                          "text": p["text"], "speaker_id": p["speaker_id"],
                          "chapter_id": p["chapter_id"], "id": p["id"], 
                          "content": "", "expected_response": ""}
                processed_data.append(prompt)
                progress_bar.update(1)
            progress_bar.close()
            return processed_data
        else:
            print(f"{self.dataset} is currently not recognized by the framework...")
            raise
    
    def select_prompts_sample(self):
        # We filter the dataset to narrow the amount of prompts(selecting scores accordingly to
        # what is defined in the parameters)
        print(f"Selecting randomized samples from \"{self.dataset_subset}\" subset")
        if "ultrafeedback_binarized" in pipe_param.dataset_name:
            filtered_dataset = self.dataset.filter(lambda example: example["score_chosen"] >= pipe_param.score_base)
        elif "librispeech" in pipe_param.dataset_name:
            filtered_dataset = self.dataset.filter(lambda example: int(example["speaker_id"]) >= 1000)
        filtered_dataset = filtered_dataset.shuffle()  #shuffled to randomize it
        random_sample = filtered_dataset.select(range(pipe_param.num_prompts))
        return self.process_dataset_format(random_sample)

