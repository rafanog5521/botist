import os.path
import jiwer
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
from config.parameters import PipelineParams
param = PipelineParams()

class Reporter:
    def __init__(self, results):
        self.report_directory = os.path.dirname(os.path.realpath(__file__))

    def create_report_folder(self):
        new_folder = self.report_directory.replace("experiments", "reports")
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

    def process_results(self, output):
        rep_folder = self.create_report_folder()
        self.snapshot_file(param.questions_path, rep_folder)
        self.snapshot_file(param.parameters_path, rep_folder)

    def process_questionnaire(self, output_list, dataset_path=param.dataset_path):
        print("Loading dataset...")
        dataset_full = load_dataset(dataset_path)
        progress_bar = tqdm(total=len(output_list), desc="Processing questionnaire...")
    
    def calculate_wer(self, output_list, dataset_path):

        # Transformación para la normalización de los textos
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.Strip(),
        ])

        # Calcula el WER usando jiwer
        wer_score = jiwer.wer(
            reference_texts,
            model_outputs,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )

        return wer_score
