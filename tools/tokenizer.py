from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from src.reports_module import Reporter
from tqdm import tqdm
import json
import torch
import argparse
import librosa
import logging

# Configurar el registro de logging para omitir mensajes de advertencia
MODEL = "facebook/wav2vec2-base-960h"
   
def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)  # Cargar audio y muestrear a 16kHz
    return audio

def get_tokens_from_audio(audio_path):
    audio = load_audio(audio_path)
    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Tamaño de batch 1

    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1) # Obtener los IDs de los tokens predichos

    return predicted_ids

def main():
    parser = argparse.ArgumentParser(description=f"Tool to tokenize audio dataset using {MODEL}")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("-s", "--subset", type=str, required=True, help="Subset of the dataset to load")
    parser.add_argument("-p", "--split", type=str, required=True, help="Split of the dataset to load (e.g., train, validation, test)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    global processor  # Make processor global so it can be accessed in subprocesses
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    global model
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)

    ds = load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
    print(f"\n* Workin with {MODEL} to tokenize...")
    progress_bar = tqdm(total=len(ds), desc="Generating tokens:")
    token_list = []
    for d in ds:
        e_text = d["text"]
        audio_path = d["audio"]["path"]
        tokens = get_tokens_from_audio(audio_path)
        raw_tokens = json.dumps(tokens[0].numpy().tolist())
        json_entry = {
            "text": e_text,
            d["id"]: raw_tokens
        }
        token_list.append(json_entry)
        progress_bar.update(1)
        if args.verbose:
            print(f"\n{json_entry}")

    progress_bar.close()
    rep = Reporter(args)
    rep.dump_info(token_list, "tokenized_output", rep.create_report_folder())


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
