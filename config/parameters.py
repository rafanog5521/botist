import os, torch

# General configuration
# pipeline values
class PipelineParams:
    def __init__(self):
        self.task = "text-generation"
        self.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        #self.model = "Phi"
        self.torch_dtype = torch.bfloat16
        self.device_map = "auto"

# Models
# prompt values for TinyLlama
class TinyLlamaParameters:
    def __init__(self):
        self.tokenize = False
        self.add_generation_prompt = True
        self.num_return_sequences = 1  # this is the key value to control the amount of possible responses obtained
        self.dataset = "HuggingFaceH4/ultrafeedback_binarized"
        self.dataset_subset = "train_prefs"
        # interaction values
        self.max_new_tokens = 1024
        self.do_sample = True
        self.temperature = 1e-32
        self.top_k = 50
        self.top_p = 0.95
# prompt values for Phi
class PhiParameters:
    def __init__(self):
        self.max_length = 512
        self.trust_remote_code=True
        self.return_attention_mask=False
#

# file routes
pipe_param = PipelineParams()
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
models_path = root_dir + "/models/" + pipe_param.model # If you have a local folder with the model
datasets_path = root_dir + "/data"
questions_path = datasets_path + "/test-questions/questions.json"  # questions file
parameters_path = datasets_path + "parameters.py"  # parameters file
num_prompts = 5
score_base = 9
