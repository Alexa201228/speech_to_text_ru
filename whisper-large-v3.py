import sys
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from utils import convert_mp3_to_wav_with_new_sample_rate, load_audio_wav

# Load model and processor globally to avoid reloading in each function call
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")

def split_audio(audio, samplerate, chunk_duration=60):
    chunk_size = chunk_duration * samplerate
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

def transcribe_chunk(args):
    chunk, samplerate = args
    # Ensure the audio is in float32 format
    chunk = chunk.astype(np.float32)
    input_features = processor(chunk, sampling_rate=samplerate, return_tensors="pt").input_features

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def transcribe_chunks_in_parallel(chunks, samplerate, max_workers=cpu_count()):
    transcriptions = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(transcribe_chunk, (chunk, samplerate)) for chunk in chunks]
        for future in futures:
            transcriptions.append(future.result())
    return " ".join(transcriptions)

def main(mp3_filename):
    # Здесь можно добавить доп папку к пути, например 
    if not os.path.exists(f"audios/{mp3_filename}"):
        print(f"Error: File {mp3_filename} does not exist.")
        sys.exit(1)

    wav_filename = "resampled_audio.wav"
    new_sample_rate = 16000  # Desired sample rate

    # Convert MP3 to WAV with the new sample rate
    convert_mp3_to_wav_with_new_sample_rate(mp3_filename, wav_filename, new_sample_rate)

    # Load and split the WAV audio
    audio, samplerate = load_audio_wav(wav_filename)
    chunks = split_audio(audio, samplerate)

    # Measure the duration of the transcription process
    start_time = time.time()
    transcription = transcribe_chunks_in_parallel(chunks, samplerate)
    end_time = time.time()

    # Print the transcription and the duration
    print("Duration of transcription process:", end_time - start_time, "seconds")

    # Write the transcription to a text file
    transcription_filename = os.path.splitext(mp3_filename)[0] + "_transcription.txt"
    with open(transcription_filename, "w") as f:
        f.write(transcription)

    print(f"Transcription saved to {transcription_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 run.py <input_mp3_file>")
        sys.exit(1)
    
    mp3_filename = sys.argv[1]
    main(mp3_filename)
    """
    Для работы скрипта необходимо установить ffmpeg в систему:

        Ubuntu/Debian:
        sudo apt-get install ffmpeg

        macOS (using Homebrew):
        brew install ffmpeg
        
        Windows:

        Загрузить ffmpeg с сайта FFmpeg website (https://ffmpeg.org/download.html).
        Распаковать файлы.
        Добавить папку с ffmpeg.exe в системные переменные PATH.

    Также необходимо устрановить зависимости из файла requirements.txt в окружение проекта
    """
