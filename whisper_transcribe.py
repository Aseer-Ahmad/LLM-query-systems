# pip install --upgrade pip
# pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

# install ffmpeg if needed

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pytube import YouTube, Channel
import pandas as pd

df = pd.read_csv(r'C:\Users\Aseer\Desktop\naqshiSaar\SHAAAMD_q(1).csv' )
video_url_list  = df.loc[: , 'video_urls']
df['trans'] = None

count = 1
for url in video_url_list:
    path = YouTube(url).streams.filter(only_audio=True)[0].download(filename=f"audio_{i}.mp3")
    print(f'audio download at : {path}')
    count+=1


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



for i in range(count):
    inp_file = f"./audio_{i}.mp3"
    result = pipe(inp_file, return_timestamps=True, generate_kwargs={"language": "english", "task": "translate"})

print(result["text"])
# print(result["chunks"])