
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from reports_module import Reporter
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
    # Cargar el archivo de audio
    audio = load_audio(audio_path)

    # Tokenizar el audio
    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Tamaño de batch 1

    # Obtener los logits del modelo
    with torch.no_grad():
        logits = model(input_values).logits

    # Obtener los IDs de los tokens predichos
    predicted_ids = torch.argmax(logits, dim=-1)

    return predicted_ids

def main():
    parser = argparse.ArgumentParser(description="Tokenize audio dataset using Wav2Vec2")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("-s", "--subset", type=str, required=True, help="Subset of the dataset to load")
    parser.add_argument("-p", "--split", type=str, required=True, help="Split of the dataset to load (e.g., train, validation, test)")
    parser.add_argument("-e", "--examples", type=int, required=False, help="")
    parser.add_argument("-n", "--num_proc", type=int, required=False, help="Split the download work to speed up the dataset processing")
    # parser.add_argument("-o", "--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    global processor  # Make processor global so it can be accessed in subprocesses
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    global model
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)

    print(f"\t * Workin with {MODEL} to tokenize...")
    ds = load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
    for d in ds:
        e_text = d["text"]
        print(f"\n\t* Text from example: >> {e_text}")
        audio_path = d["audio"]["path"]
        tokens = get_tokens_from_audio(audio_path)
        print(tokens)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
