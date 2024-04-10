import os.path
from datetime import datetime
from config import parameters


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
        self.snapshot_file(parameters.questions_path, rep_folder)
        self.snapshot_file(parameters.parameters_path, rep_folder)

    def process_answers(self, output):
        for q in output:  # this iterates over the list of questions and their respective answers
            vector_distance = 0
            for answers in q["answers"]:  # this iterates over the answers list
                vector_distance += self.compare_answers(answers)
            output["comparison"] = vector_distance / len(answers)

    def compare_answers(self, answers):
        for answer in range(len(a)):
            for possibility in range(len(a)):
                if possibility != answer:
                    vector_distance += self.levenshtein_distance(answer, possibility)

    def levenshtein_distance(self, a1, a2):
        # this function measures the distance between one answer and another applying the levenshtein distance edition
        # measurement which considers the minimum changes that need to be made in a string to transform it into another
        matrix = [[0] * (len(a2) + 1) for _ in range(len(a1) + 1)]

        # Inicializar la primera fila y la primera columna
        for i in range(len(a1) + 1):
            matrix[i][0] = i
        for j in range(len(a2) + 1):
            matrix[0][j] = j

        # calculate levenshtein distance
        for i in range(1, len(a1) + 1):
            for j in range(1, len(a2) + 1):
                cost = 0 if a1[i - 1] == a2[j - 1] else 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

        # return levenshtein distance between the two strings
        return matrix[len(a1)][len(a2)]
