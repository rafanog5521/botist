######
# Dado un archivo, verificar y calcular el WER de la tokenizacion de un dataset tokenizado
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import argparse
import logging
import json

MODEL = "facebook/wav2vec2-base-960h"

def decode_tokens(rosetta, token_array):
    for t in token_array:
        translation = []
        print(f"Decoding: {t}")
        for token in t.values:
            print(token)
        input()

def decode_rosetta(stone):
    inverted_dict = {}
    for key, value in stone.items():
        if value not in inverted_dict:
            inverted_dict[value] = key
        else:
            inverted_dict[value].append(key)
    return inverted_dict

def main():
    parser = argparse.ArgumentParser(description=f"Tool to tokenize audio dataset using {MODEL}")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("-s", "--subset", type=str, required=True, help="Subset of the dataset to load")
    parser.add_argument("-p", "--split", type=str, required=True, help="Split of the dataset to load (e.g., train, validation, test)")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to file with the tokens")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    rosetta = decode_rosetta(Wav2Vec2Processor.from_pretrained(MODEL).tokenizer.vocab)
    raw_ds = load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
    with open(args.file, "r") as file:
        tokens = json.load(file)
    
    decode_tokens(rosetta, tokens)
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
