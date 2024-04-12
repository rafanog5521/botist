from config import parameters
import transformers
from transformers import pipeline
import time
import jiwer
import argparse


class TinyLlamaModelInteractor():
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
        prompt = self.prompt(question)
        self.__pipe__(prompt, max_new_tokens=parameters.max_new_tokens, do_sample=parameters.do_sample,temperature=parameters.temperature, top_k=parameters.top_k, top_p=parameters.top_p)

    def ask_question(self, question):
        prompt = self.prompt(question)
        output =  self.__pipe__(prompt, max_new_tokens=parameters.max_new_tokens, do_sample=parameters.do_sample,
                                temperature=parameters.temperature, top_k=parameters.top_k, top_p=parameters.top_p)

        return output[0]["generated_text"]  # .split('<|assistant|>\n')

