import torch
import os

# pipeline values
task = "text-generation"
model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset = "/TinyLlama-test-questions/questions.json"
torch_dtype = torch.bfloat16
device_map = "auto"

# file routes
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
models_path = root_dir + "/models/" + model # If you have a local folder with the model
datasets_path = root_dir + "/datasets"
questions_path = datasets_path + dataset  # questions file
parameters_path = datasets_path + "parameters.py"  # parameters file

###############################
# prompt values
tokenize = False
add_generation_prompt = True
num_return_sequences = 1  # this is the key value to control the amount of possible responses obtained
# interaction values
max_new_tokens = 1024
do_sample = True
temperature = 1e-32
top_k = 50
top_p = 0.95
###############################
