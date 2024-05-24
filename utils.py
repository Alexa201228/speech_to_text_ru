import librosa
import numpy as np
from pydub import AudioSegment
import sounddevice as sd
from scipy.io.wavfile import write, read


# Function to record audio from the microphone
def record_audio(duration=5, filename="output.wav", samplerate=16000):
    print("Recording...")
    myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Finished recording.")
    write(filename, samplerate, myrecording)  # Save as WAV file


# Load the audio file
def load_audio(filename):
    samplerate, data = read(filename)
    return data, samplerate

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