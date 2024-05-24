# Load model directly
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from utils import load_audio

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

filename = "output.wav"
audio, samplerate = load_audio(filename)

# Ensure the audio is in float32 format
audio = audio.astype(np.float32) / np.iinfo(np.int16).max

# Process the audio using the processor
input_values = processor(audio, sampling_rate=samplerate, return_tensors="pt").input_values

# Generate transcription
with torch.no_grad():
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)