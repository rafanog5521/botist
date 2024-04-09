import torch
import os


# file routes
current_dir = os.path.dirname(os.path.realpath(__file__))
questions_path = current_dir + "/questions.txt"  # questions file
parameters_path = current_dir + "/parameters.py"  # parameters file

# pipeline values
task = "text-generation"
model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
torch_dtype = torch.bfloat16
device_map = "auto"
num_return_sequences = 5  # this is the key value to control the amount of possible responses obtained

###############################
# prompt values
tokenize = False
add_generation_prompt = True
# interaction values
max_new_tokens = 256
do_sample = True
temperature = 0.7
top_k = 50
top_p = 0.95

###############################
