from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from src.reports_module import Reporter
from tqdm import tqdm
import json
import torch
import argparse
import librosa
import logging
import concurrent.futures

MODEL = "facebook/wav2vec2-base-960h"
SAMPLING_RATE = 16000

def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)  # Sampling rate = 16kHz
    return audio

def get_tokens_from_audio(audio_path):
    audio = load_audio(audio_path)
    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=SAMPLING_RATE).input_values.to(device)  
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)

    return predicted_ids

def process_entry(d):
    e_text = d["text"]
    audio_path = d["audio"]["path"]
    tokens = get_tokens_from_audio(audio_path)
    raw_tokens = json.dumps(tokens[0].cpu().numpy().tolist()) 
    json_entry = {
        "text": e_text,
        d["id"]: raw_tokens
    }
    return json_entry

def main():
    parser = argparse.ArgumentParser(description=f"Tool to tokenize audio dataset using {MODEL}")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("-s", "--subset", type=str, required=True, help="Subset of the dataset to load")
    parser.add_argument("-p", "--split", type=str, required=True, help="Split of the dataset to load (e.g., train, validation, test)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads to use for parallel processing")  
    args = parser.parse_args()

    global processor  # Both processor and model need to be global so that it can be accessible for subprocess
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    global model
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
    print(f"\n* Working with {MODEL} to tokenize...")
    progress_bar = tqdm(total=len(ds), desc="Generating tokens:")
    token_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_entry, d): d for d in ds}
        for future in concurrent.futures.as_completed(futures):
            try:
                json_entry = future.result()
                token_list.append(json_entry)
                if args.verbose:
                    print(f"\n{json_entry}")
            except Exception as exc:
                print(f"Generated an exception: {exc}")
            progress_bar.update(1)

    progress_bar.close()
    rep = Reporter(args)
    rep.dump_info(token_list, "tokenized_output", rep.create_report_folder())

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
