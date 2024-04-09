import parameters
import transformers
from transformers import pipeline


class ModelInteractor:
    def __init__(self):
        self.__pipe__ = pipeline(task=parameters.task, model=parameters.model,
                                 torch_dtype=parameters.torch_dtype, device_map=parameters.device_map,
                                 num_return_sequences=parameters.num_return_sequences)
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)  # disable base warnings

    def prompt(self, question):
        new_q = [question]
        return self.__pipe__.tokenizer.apply_chat_template(new_q, tokenize=parameters.tokenize,
                                                           add_generation_prompt=parameters.add_generation_prompt)

    def ask_question(self, question):
        result = {"question": question["content"]}
        prompt = self.prompt(question)
        outputs = self.__pipe__(prompt, max_new_tokens=parameters.max_new_tokens, do_sample=parameters.do_sample,
                                temperature=parameters.temperature, top_k=parameters.top_k, top_p=parameters.top_p)

        answers = []
        index = 0
        for a in outputs:
            answers.append(((outputs[index]['generated_text']).split('<|assistant|>\n'))[1])
            index += 1

        result["answers"] = answers
        return result
