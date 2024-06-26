from datasets import load_dataset
from config.parameters import TinyLlamaParameters, PhiParameters, WhisperParameters
import transformers
import os, time
import re
from tqdm import tqdm
#Tinyllama
from transformers import pipeline
#Phi
from transformers import AutoModelForCausalLM, AutoTokenizer
#Whisper
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
import torch
from evaluate import load
import random
import numpy as np

class TinyLlamaModelInteractor:
    def __init__(self, pipeline_params):
        self.pipe_param = pipeline_params
        self.tiny_param = TinyLlamaParameters()
        self.__pipe__ = pipeline(task=self.pipe_param.task, model=self.pipe_param.model,
                                 torch_dtype=self.pipe_param.torch_dtype, device_map=self.pipe_param.device_map,
                                 num_return_sequences=self.tiny_param.num_return_sequences)
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)  # disable base warnings
        self.dataset = self.tiny_param.dataset
        self.dataset_subset = self.tiny_param.dataset_subset
        self.dataset_split = self.tiny_param.dataset_split
        torch.set_default_device("cuda")
        SEED=0
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    def prompt(self, question):
        SYSTEM_PROMPT = {
            "role": "system",
            "content": "You are a friendly chatbot who tries to provide short and meaningful answers.",
        }
        prompt = [SYSTEM_PROMPT, question]
        return self.__pipe__.tokenizer.apply_chat_template(prompt, tokenize=self.tiny_param.tokenize,
                                                           add_generation_prompt=self.tiny_param.add_generation_prompt)

    def init_model(self, question='Say Hello'): #Use a sample question to trigger downloads for Tinyllama resources.
        self.prompt(question)

    def ask_question(self, question, performance_metric=True):

        prompt = self.prompt(question)
        if performance_metric:
            start_time = time.time()

        output =  self.__pipe__(prompt, max_new_tokens=self.tiny_param.max_new_tokens, do_sample=self.tiny_param.do_sample,
                                temperature=self.tiny_param.temperature, top_k=self.tiny_param.top_k, top_p=self.tiny_param.top_p)
        try:
            response = {"output": output[0]["generated_text"].split('<|assistant|>\n')[1]}
        except:
            response = {"output": output[0]["generated_text"]}

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
    def __init__(self, pipeline_params):
        self.pipe_param = pipeline_params
        self.phi_param = PhiParameters()
        self.model = AutoModelForCausalLM.from_pretrained(self.pipe_param.model, torch_dtype=self.pipe_param.torch_dtype, trust_remote_code=self.phi_param.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pipe_param.model, trust_remote_code=self.phi_param.trust_remote_code)
        self.dataset = self.phi_param.dataset
        self.dataset_subset = self.phi_param.dataset_subset
        self.dataset_split = None
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
    def __init__(self, pipeline_params):
        self.pipe_param = pipeline_params
        self.whisper_param = WhisperParameters()
        self.dataset = self.whisper_param.dataset
        self.dataset_subset = self.whisper_param.dataset_subset
        self.dataset_split = self.whisper_param.dataset_split
        if not hasattr(self.whisper_param, "audio_folder"):
            self.dataset_loaded = load_dataset(self.whisper_param.dataset, self.dataset_subset, split=self.dataset_split, trust_remote_code=True)
        else: # will use a defined dataset not loaded through the dataset library
            self.audio_folder = self.whisper_param.audio_folder # folder containing the wav files
            self.reference_file = self.whisper_param.reference_file # file containing the original references

        self.model = WhisperForConditionalGeneration.from_pretrained(self.pipe_param.model).to("cuda")
        self.processor = WhisperProcessor.from_pretrained(self.pipe_param.model)

    def init_model(self):
        pass

    def map_to_pred(self, batch):
        audio = batch["audio"]
        input_features = self.processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        batch["reference"] = self.processor.tokenizer._normalize(batch['content'])

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features.to("cuda"))[0]
        transcription = self.processor.decode(predicted_ids)
        batch["prediction"] = self.processor.tokenizer._normalize(transcription)
        return batch

    def evaluate_speech(self):
        result = self.dataset_loaded.map(self.map_to_pred)
        return (result, load)

    def transcription_of_speech(self, speech, performance_metric=True):
        if performance_metric:
            start_time = time.time()
        if not hasattr(self, 'audio_folder') and not hasattr(self, 'references_folder'):
            sample = speech["audio"]
            input_features = self.processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
        else:
            audio, sampling_rate = sf.read(speech["audio"]) # read the audio file
            input_features = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

        # generate token ids
        predicted_ids = self.model.generate(input_features.to("cuda"))[0]
        # decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #Format output
        readable_transcription = ','.join(map(str, transcription))
        readable_transcription = (re.sub(",", "", readable_transcription))
        if performance_metric:
            end_time = time.time()
            response_time = end_time - start_time # Calculate response time
            total_tokens_generated = len(predicted_ids) # Calculate tokens per second
            tokens_per_second = total_tokens_generated / response_time

        readable_transcription = readable_transcription.replace("<|startoftranscript|><|notimestamps|>", "").replace("<|endoftext|>", "")
        return {"current_response": readable_transcription, "response_time": response_time, "tokens_per_sec": tokens_per_second}

class DatasetInteractor():
    def __init__(self, pipeline_params, dataset, subset, subset_split):
        self.pipe_param = pipeline_params
        print("\n*\tLoading dataset...")
        if "local_audio" not in self.pipe_param.dataset_name:
            try:
                self.dataset = load_dataset(dataset, subset, split=subset_split)
            except Exception as e:
                print("Error loading dataset: {}".format(e))
                raise
            
            self.dataset_subset = subset
            self.dataset_name = dataset
        else:
            self.dataset_path = dataset

    def process_dataset_format(self, data):  # This is to standardize the format of the prompt list for report purpose
        progress_bar = tqdm(total=len(data), desc="Formatting dataset:")
        processed_data = []
        if "ultrafeedback_binarized" in self.pipe_param.dataset_name:
            for p in data:
                prompt = {"role": "user", "content": p["prompt"], "prompt_id": p["prompt_id"],
                          "expected_response": p["chosen"][1]}
                processed_data.append(prompt)
                progress_bar.update(1)
            progress_bar.close()
        elif "librispeech" in self.pipe_param.dataset_name:
            for p in data:
                prompt = {"file": p["file"], "audio": p["audio"],
                          "expected_response": p["text"], "speaker_id": p["speaker_id"],
                          "chapter_id": p["chapter_id"], "id": p["id"]}
                processed_data.append(prompt)
                progress_bar.update(1)
            progress_bar.close()
        elif "audio" in self.pipe_param.dataset_name:               
            for r in refs:
                prompt = {"expected_response": refs[r], "audio": wavs[refs.index(r)]}
                processed_data.append(prompt())
                progress_bar.update(1)
            progress_bar.close()

        else:
            print("{} is currently not recognized by the framework...".format(self.dataset))
            assert False

        return processed_data
    
    def capture_wavs(self):
        list_files = os.listdir(self.dataset_path)
        wavs = [audio for audio in list_files if audio.endswith("wav")]
        wavs_paths = [os.path.join(self.dataset_path, wav) for wav in wavs]
        return wavs_paths

    def select_prompts_sample(self):
        # We filter the dataset to narrow the amount of prompts(selecting scores accordingly to
        # what is defined in the parameters)
        if "ultrafeedback_binarized" in self.pipe_param.dataset_name:
            filtered_dataset = self.dataset
        elif "librispeech" in self.pipe_param.dataset_name:
            filtered_dataset = self.dataset
        elif "local_audio" in self.pipe_param.dataset_name:
            wavs = self.capture_wavs()
            with open(os.path.join(self.dataset_path, "references.txt"), "r") as rfile:
                refs = []
                for r in rfile:
                    refs.append(r)
            filtered_dataset = []
            for wav in wavs:
                prompt = {
                    "expected_response": refs[wavs.index(wav)],
                    "audio": wav
                }
                filtered_dataset.append(prompt)
        else:
            print("Error processing the dataset sample")
            raise ValueError

        if "local_audio" in self.pipe_param.dataset_name:
            selected_sample = filtered_dataset[:self.pipe_param.num_prompts]
            return selected_sample
        else:
            return self.process_dataset_format(filtered_dataset.select(range(self.pipe_param.num_prompts)))