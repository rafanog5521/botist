from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from tqdm import tqdm
import torch
import os


# To run these tests, there need to be wav files inside the datasets file

def capture_wavs():
    current_dir = os.path.dirname(os.path.realpath(__file__)).replace("/src", "")
    folder = os.path.join(current_dir, "datasets")
    list_files = os.listdir(folder)
    wavs = [audio for audio in list_files if audio.endswith("wav")]
    wavs_paths = [os.path.join(folder, wav) for wav in wavs]
    return wavs_paths

def transcribe_audio(audio_path):
    # read audio file
    audio, sampling_rate = sf.read(audio_path)
    # process audio using the processor
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
    return transcription

###################################################################################
print("Loading model and processor...")
# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
print("Model and processor loaded!")

# Example usage:
wavs = capture_wavs()
progress_bar = tqdm(total=len(wavs), desc="Transcribing wavs:")
transcription = []
for audio in wavs:
    transcription.append(transcribe_audio(audio).replace("<|startoftranscript|><|notimestamps|> ", "").replace("<|endoftext|>", ""))
    progress_bar.update(1)

progress_bar.close()

for text in transcription:
    print(f"transcription>>>>{text}")
