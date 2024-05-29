import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pydub import AudioSegment
import time

# Load the model and processor
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")

# Convert MP3 to WAV with a new sample rate
def convert_mp3_to_wav_with_new_sample_rate(mp3_filename, wav_filename, new_sample_rate):
    audio = AudioSegment.from_mp3(mp3_filename)
    audio = audio.set_frame_rate(new_sample_rate)
    audio.export(wav_filename, format="wav")

# Load the WAV file
def load_audio_wav(filename):
    audio = AudioSegment.from_file(filename)
    sample_rate = audio.frame_rate
    audio = np.array(audio.get_array_of_samples())
    if audio.ndim == 2:  # If stereo, convert to mono
        audio = audio.mean(axis=1)
    return audio, sample_rate

# Function to transcribe a single chunk
def transcribe_chunk(chunk, samplerate):
    # Ensure the audio is in float32 format
    chunk = chunk.astype(np.float32)
    input_features = processor(chunk, sampling_rate=samplerate, return_tensors="pt").input_features

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Paths to your MP3 and WAV files
mp3_filename = "your_audio_file.mp3"
wav_filename = "output.wav"
new_sample_rate = 16000  # Desired sample rate

# Convert MP3 to WAV with the new sample rate
# convert_mp3_to_wav_with_new_sample_rate(mp3_filename, wav_filename, new_sample_rate)

# Load and split the WAV audio
audio, samplerate = load_audio_wav(wav_filename)

# Empirically determine the maximum chunk size
max_chunk_duration = 30  # Start with a reasonable chunk duration in seconds
step_duration = 10       # Step to increase the chunk duration

while True:
    chunk_size = max_chunk_duration * samplerate
    if len(audio) < chunk_size:
        chunk_size = len(audio)
    chunk = audio[:chunk_size]

    try:
        start_time = time.time()
        transcription = transcribe_chunk(chunk, samplerate)
        end_time = time.time()
        print(f"Chunk size (duration): {max_chunk_duration} seconds")
        print(f"Transcription duration: {end_time - start_time} seconds")
        print("Transcription:", transcription)
        
        if len(audio) < chunk_size:
            break  # Reached the end of the audio
    except RuntimeError as e:
        print(f"Failed at chunk duration {max_chunk_duration} seconds: {str(e)}")
        break  # Stop if it fails due to memory or other constraints

    max_chunk_duration += step_duration

print(max_chunk_duration)
