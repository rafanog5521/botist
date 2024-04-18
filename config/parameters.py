import os, torch

# General configuration
# pipeline values
class PipelineParams:
    def __init__(self):
        #setup
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # 'microsoft/phi-2'
        self.dataset_name = "HuggingFaceH4/ultrafeedback_binarized"

        self.task = "text-generation"
        self.torch_dtype = torch.bfloat16
        self.device_map = "auto"
        self.num_prompts = 5
        self.score_base = 9

        #paths
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_dir = os.path.dirname(self.current_dir)

        #remote or local model
        self.models_path = self.root_dir + "/models/" + self.model_name # If you have a local folder with the model
        if (os.path.exists(self.models_path)):
            self.model = self.models_path
            self.local_model = True
        else: 
            self.model = self.model_name
            self.local_model = False
        
        #remote or local dataset
        self.dataset_path = self.root_dir + "/datasets/" + self.dataset_name # If you have a local folder with the dataset
        if (os.path.exists(self.dataset_path)):
            self.dataset = self.dataset_path
            self.local_dataset = True
        else: 
            self.dataset = self.dataset_name
            self.local_dataset = False

        self.questions_path = self.dataset_path + "/test-questions/questions.json"  # questions file
        self.parameters_path = self.dataset_path + "parameters.py"  # parameters file


# Models
# prompt values for TinyLlama
class TinyLlamaParameters:
    def __init__(self):
        param = PipelineParams()
        self.tokenize = False
        self.add_generation_prompt = True
        self.num_return_sequences = 1  # this is the key value to control the amount of possible responses obtained
        self.model = param.model
        self.dataset = param.dataset
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