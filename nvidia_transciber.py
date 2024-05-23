import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_ru_fastconformer_hybrid_large_pc")

transcriptions = asr_model.transcribe(["output.wav"])
print(transcriptions)