import os.path
import jiwer
import json
import matplotlib.pyplot as plt 
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
from config import parameters


class Reporter:
    def __init__(self, params):
        self.report_directory = os.path.dirname(os.path.realpath(__file__))
        self.params = params

    def create_report_folder(self):
        new_folder = self.report_directory.replace("src", "reports")
        if not os.path.exists(new_folder):
            print(f"Creating {new_folder} folder as it doesn't exists...")
            os.mkdir(new_folder)

        date_string = "{}".format(datetime.today()).replace(":", "-").replace(".", "-")
        report_folder = new_folder + "/{}_log".format(date_string)
        print(f"Creating report log here>>{report_folder}")
        os.mkdir(report_folder)
        return report_folder

    def snapshot_file(self, file, rep_path):
        print(f"Saving copy of {file}")
        with open(file, "rb") as file_to_snapshot:
            content = file_to_snapshot.read()

        snapshot_path = os.path.join(rep_path, os.path.basename(file))
        with open(snapshot_path, "wb") as result:
            result.write(content)

    def process_results(self, output, debug=False):
        print("Preparing report folder")
        rep_folder = self.create_report_folder()
        if debug:
            print("Generating raw output")
            self.dump_info(output, "raw_results", rep_folder) # creates a raw copy of the results
        
        expected, current, response_time, tps = self.process_questionnaire(output) # parse results

        wer = self.calculate_wer(expected, current)
        self.graphicate_results(range(1, len(response_time) + 1), response_time, "Question Num.", "Time", 
                                "Response time", rep_folder, debug)
        self.graphicate_results(range(1, len(tps) + 1), tps, "Question Num.", "# Tokens/S", "Tokens per second", 
                                rep_folder, debug)

        print("Generating summary file")
        summary = {
            "parameters": {
                "model": self.params.model
            },
            "num_prompts_used": len(output),
            "WER": wer,
            "average_resp_time": sum(response_time) / len(response_time),
            "average_tokens_per_seccond": sum(tps) / len(tps)
        }
        self.dump_info(summary, "results_summary", rep_folder)
        print(f"Reports can be viewed at: {rep_folder}")

    def dump_info(self, output, name, path):
        with open(os.path.join(path, f"{name}.json"), "w") as file: # the as needs to be changed as it superseeds the dependency
            json.dump(output, file, indent=4)

    def process_questionnaire(self, output_list):
        expected_responses = []
        current_responses = []
        responses_times = []
        tokens_per_second = []
        progress_bar = tqdm(total=len(output_list), desc="Parsing responses for reports:")
        for r in output_list:
            expected_responses.append(r["expected_response"]["content"])
            current_responses.append(r["response"]["output"])
            responses_times.append(r["response"]["response_time"])
            tokens_per_second.append(r["response"]["tokens_per_second"])
            progress_bar.update(1)
        progress_bar.close()
        return expected_responses, current_responses, responses_times, tokens_per_second
        
    def calculate_wer(self, reference_texts, model_outputs):
        print("Calculating WER")
        transforms = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])

        wer = jiwer.wer(reference_texts, model_outputs,truth_transform=transforms,hypothesis_transform=transforms)

        return wer

    def graphicate_results(self, x, y, x_desc, y_desc, title, file_path, display=False):
        print(f"Generating {title} graphic")
        fig, ax = plt.subplots()
        ax.plot(x, y, label="placeholder example")
        ax.set_title(title)
        ax.set_xlabel(x_desc)
        ax.set_ylabel(y_desc)
        ax.legend()
        if display:
            plt.show()
        graph_path = os.path.join(file_path, title + ".pdf") 
        plt.savefig(graph_path)
        print(f"Saved {title} in {graph_path}")