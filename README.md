# BOTIST
Test framework to evaluate WER and Performance for different IA models and versions


## Setup (local)
- Put models on ./models folder
- Put datasets on ./datasets folder
- Set python virtual enviroment (optional)
  - python3 -m venv env
  - source ./env/bin/activate
- Run pip install -r ./requirements.txt
- Run export PYTHONPATH="${PYTHONPATH}:/yourfolders/botist/"
- ./src/run_model.py

## Setup (docker, remote model/dataset)
- Check config/parameters.py if model and dataset are correct
- Run "make build_docker_model" build basic docker file and download dependencies for your model
- Run "make run_model" to run the model with your remote model/datasets

## Setup (docker, local model/dataset)
- Download model and/or dataset to a folder
- Check config/parameters.py if model and dataset are correct
- parameters.py model and dataset definition should match folder structure passed on CLI
  - Example: "home/user/Desktop/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0" and "model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"" on the python file # 
  - Same applies for datasets
- Run "make build_docker_model m=model_path d=dataset_path" to build basic docker file and download dependencies for your model
- Run "make run_model m=model_path d=dataset_path" to run the model with your local model/datasets

## Measurements
## WER
There are currently 2 applications of this metric within the code:
  - 5 different inputs: To show average WER within different topics.
  - 1 input sent 5 times: To show WER consistency when the model receives a particular input several times.

## Performance
Performance metric is applied within the following contexts:
  - We meassure the time that it takes the model to generate the first token.
  - Calculate the amount of tokens generated per second.
