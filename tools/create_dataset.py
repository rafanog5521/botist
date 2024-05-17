#!/usr/bin/env python3
#Example of usage:
#./tools/create_dataset.py --input_path /root/botist/datasets/custom/wer_calculated_6k.json --dataset_name librispeech_asr --dataset_subset clean --dataset_split train.360
import json, argparse
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('--input_path', help='Where is located the json file to convert?', default="/root/botist/datasets/custom/wer_calculated_output_6k", required=True, type=str)
parser.add_argument('--dataset_name', help='Dataset to be filtered', required=True, type=str)
parser.add_argument('--dataset_subset', help='Subset of the dataset', required=True, type=str)
parser.add_argument('--dataset_split', help='Split of the dataset', required=True, type=str)
parser.add_argument('--output_path', help='Folder to save the dataset on ./dataset', default="/root/botist/datasets/custom", required=False, type=str)
parser.add_argument('--WER_threshold', help='Threshold', required=False, default=0.1, type=float)
args = vars(parser.parse_args())

# Read the JSON file
with open(args["input_path"], 'r') as file: data = json.load(file)

# Filter IDs by WER value
filtered_ids = sorted([item["id"] for item in data if item.get('WER', float('inf')) <= args["WER_threshold"]])

#Load dataset
dataset = load_dataset(args["dataset_name"], args["dataset_subset"], split=args["dataset_split"])
#Filter by IDs
filtered_dataset = dataset.filter(lambda ds: ds['id'] in filtered_ids)
print(filtered_dataset)
filtered_dataset.save_to_disk(args["output_path"])
print("Saved dataset on: ", args['output_path'])