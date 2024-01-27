# pip install --upgrade pip
# pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

# pip install flash-attn --no-build-isolation
# - model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
# + model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True)


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline:
# - result = pipe(sample)
# + result = pipe("audio.mp3")

# Whisper predicts the language of the source audio automatically. If the source audio language is known a-priori,
# it can be passed as an argument to the pipeline:
# result = pipe(sample, generate_kwargs={"language": "english"})

inp_file = None
result = pipe(inp_file, return_timestamps=False)
print(result["text"])
