#!/usr/bin/env python3
import os, json, argparse
from config.parameters import *
from reports_module import Reporter

# Parameter section
parser = argparse.ArgumentParser(description='Run model')
parser.add_argument("--folders", nargs="*")
args = vars(parser.parse_args())
folders = args["folders"]

model = None
with open("./"+folders[0]+"/results_summary.json", 'r') as summary:
    file = json.load(summary)
    model = file["parameters"]["model"]

with open("./"+folders[0]+"/raw_results.json", 'r') as reference:
    reference_data = json.load(reference)

    folders.pop(0)
    index=0
    for folder in folders:
        WER = []
        with open("./"+folder+"/raw_results.json", 'r') as file:
            data = json.load(file)

            while(index<len(data)):
                print("\n*\t{}".format(index))
                if 'whisper' in model.lower():
                    prompt = reference_data[index]["expected_response"]
                elif 'tinyllama' in model.lower():
                    prompt = reference_data[index]["content"]
                prompt = prompt.split()
                prompt = ' '.join(prompt[:30])
                print("*\t{}...".format(prompt))
                r = Reporter
                if 'whisper' in model.lower():
                    line_wer = r.calculate_wer_per_line(Reporter, reference_data[index]["response"], data[index]["response"])
                elif 'tinyllama' in model.lower():
                    line_wer = r.calculate_wer_per_line(Reporter, reference_data[index]["response"]["output"], data[index]["response"]["output"])
                print("*\tWER: {:.2f}".format(line_wer))
                WER.append(line_wer)
                index=index+1

        total = len(data)
        equals = WER.count(0.0)
        percentage = (equals/total)*100
        print("\n\n*\tOf {} lines, {} are equal. That is a {:.2f}% coincidence.\n".format(total, equals, percentage))
