from config import parameters
import transformers
from transformers import pipeline
import time
import jiwer


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

