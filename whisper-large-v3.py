# Load model directly
import time
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from utils import convert_mp3_to_wav_with_new_sample_rate, load_audio_wav

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")

# Split audio into smaller chunks
def split_audio(audio, samplerate, chunk_duration=30):
    chunk_size = chunk_duration * samplerate
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

# Preprocess and transcribe audio chunks
def transcribe_chunks(chunks, samplerate):
    transcriptions = []
    for chunk in chunks:
        # Ensure the audio is in float32 format
        chunk = chunk.astype(np.float32) / np.iinfo(np.int16).max
        input_features = processor(chunk, sampling_rate=samplerate, return_tensors="pt").input_features

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        # Decode the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)
    return " ".join(transcriptions)

# Paths to your MP3 and WAV files
mp3_filename = "test_dialog.mp3"
wav_filename = "test_dialog_resampled.wav"
new_sample_rate = 16000  # Desired sample rate

# Convert MP3 to WAV with the new sample rate
convert_mp3_to_wav_with_new_sample_rate(mp3_filename, wav_filename, new_sample_rate)

# Load and split the WAV audio
audio, samplerate = load_audio_wav(wav_filename)
chunks = split_audio(audio, samplerate)

# Measure the duration of the transcription process
start_time = time.time()
transcription = transcribe_chunks(chunks, samplerate)
end_time = time.time()

# Print the transcription and the duration
print("Transcription:", transcription)
print("Duration of transcription process:", end_time - start_time, "seconds")

with open("test_transcription.txt", "w") as f:
    f.write(transcription)
    