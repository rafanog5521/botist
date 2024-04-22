# BOTIST
Test framework to evaluate WER and Performance for different IA models and versions

## Main functioning:
![Functional Diagram](image.png)


## Setup
- Put models on ./models folder
- Put datasets on ./datasets folder
- Set python virtual enviroment (optional)
  - python3 -m venv env
  - source ./env/bin/activate
- Run pip install -r ./requirements.txt
- Run export PYTHONPATH="${PYTHONPATH}:/yourfolders/botist/"
- ./src/run_models.py

## Measurements
## WER
There are currently 2 applications of this metric within the code:
  - 5 different inputs: To show average WER within different topics.
  - 1 input sent 5 times: To show WER consistency when the model receives a particular input several times.

## Performance
Performance metric is applied within the following contexts:
  - We meassure the time that it takes the model to generate the first token.
  - Calculate the amount of tokens generated per second.
