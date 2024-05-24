from scipy.io.wavfile import read
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np

from utils import load_audio, record_audio

# Load the model and processor
processor = WhisperProcessor.from_pretrained("mitchelldehaven/whisper-large-v2-ru")
model = WhisperForConditionalGeneration.from_pretrained("mitchelldehaven/whisper-large-v2-ru")



# Uncomment this line if you need to record audio and set recording duration in seconds
# record_audio(duration=60)

# Preprocess the audio
filename = "output.wav"
audio, samplerate = load_audio(filename)

# Ensure the audio is in float32 format
audio = audio.astype(np.float32) / np.iinfo(np.int16).max

# Process the audio using the processor
input_features = processor(audio, sampling_rate=samplerate, return_tensors="pt").input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)
