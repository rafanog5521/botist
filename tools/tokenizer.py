from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from src.reports_module import Reporter
from tqdm import tqdm
import json
import torch
import argparse
import torchaudio
import librosa
import logging
import concurrent.futures
import soundfile as sf

MODEL = "facebook/wav2vec2-base-960h"
SAMPLING_RATE = 16000
EXTRACTED_PATH_FRACTION = "/LibriSpeech/dev-clean/" # this is to cover the difference between the theoretical path and actual path after extraction

def load_audio(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)
        return waveform.squeeze().numpy()
    except Exception as e:
        logging.error(f"torchaudio failed to load {audio_path} with error: {e}")
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
            return audio
        except Exception as e:
            logging.error(f"librosa also failed to load {audio_path} with error: {e}")
            try:
                audio, sr = sf.read(audio_path)
                if sr != SAMPLING_RATE:
                    audio = librosa.resample(audio, sr, SAMPLING_RATE)
                return audio
            except Exception as e:
                logging.error(f"soundfile also failed to load {audio_path} with error: {e}")
                raise

def get_tokens_from_audio(audio_info, load_audio=False):
    if load_audio:
        audio = load_audio(audio_info)
    else:
        audio = audio_info
    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=SAMPLING_RATE).input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return predicted_ids

def process_entry(entry, args):
    results = []
    try:
        if not all(key in entry for key in ["text", "audio"]):
            raise KeyError(f"Missing keys in entry: {entry}")
        ## this next snippet need to be implemented due to a recent change in the structure how the dataset is extracted
        if args.load_audio:
            audio_path = entry["audio"]["path"]
            entry_id_path = EXTRACTED_PATH_FRACTION + str(entry["speaker_id"]) + "/" + str(entry["chapter_id"])
            audio_info = audio_path[:audio_path.rfind("/")] + entry_id_path + audio_path[audio_path.rfind("/"):]
        else:
            audio_info = entry["audio"]["array"]
        tokens = get_tokens_from_audio(audio_info)
        raw_tokens = tokens[0].cpu().numpy().tolist()
        json_entry = {
            entry["id"]: raw_tokens
        }
        results.append(json_entry)
    except Exception as e:
        logging.error(f"Exception: {e} in entry {entry}")
    return results

def process_ds(args, ds):
    progress_bar = tqdm(total=len(ds), desc="Generating tokens:")
    token_list = []

    num_batches = (len(ds) + args.batch_size - 1) // args.batch_size 
    for i in range(num_batches):
        batch_start = i * args.batch_size
        batch_end = min((i + 1) * args.batch_size, len(ds))
        batch = ds[batch_start:batch_end]

        # To reconstruct the dictionaries
        batch_entries = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_entry, entry, args) for entry in batch_entries]
            for future in concurrent.futures.as_completed(futures):
                try:
                    json_entries = future.result()
                    if json_entries:
                        token_list.extend(json_entries)
                        if args.verbose:
                            for entry in json_entries:
                                print(f"\n{entry}")
                except Exception as exc:
                    logging.error(f"Generated an exception: {exc}")

            progress_bar.update(len(batch_entries))

    progress_bar.close()
    return token_list

def main():
    parser = argparse.ArgumentParser(description=f"Tool to tokenize audio dataset using {MODEL}")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("-s", "--subset", type=str, required=True, help="Subset of the dataset to load")
    parser.add_argument("-p", "--split", type=str, required=True, help="Split of the dataset to load (e.g., train, validation, test)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-l", "--load-audio", action="store_true", help="Indicates to work from the audio file instead of the array")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads to use for parallel processing")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    args = parser.parse_args()

    global processor
    processor = Wav2Vec2Processor.from_pretrained(MODEL, ignore_mismatched_sizes=True)
    global model
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, ignore_mismatched_sizes=True)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
    
    print(f"\n* Working with {MODEL} to tokenize...")

    token_list = process_ds(args, ds)
    rep = Reporter(args)
    rep.dump_info(token_list, args.subset + "_" + args.split, rep.create_report_folder())

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
