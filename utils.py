import sounddevice as sd
from scipy.io.wavfile import write, read


# Function to record audio from the microphone
def record_audio(duration=5, filename="output.wav", samplerate=16000):
    print("Recording...")
    myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Finished recording.")
    write(filename, samplerate, myrecording)  # Save as WAV file


