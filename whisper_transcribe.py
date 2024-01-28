# pip install --upgrade pip
# pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

# install ffmpeg if needed

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pytube import YouTube, Channel
import pandas as pd

df = pd.read_csv(r'C:\Users\Aseer\Desktop\naqshiSaar\LLM-query-systems\SHAAAMD_q(1).csv' )
video_url_list  = df.loc[: , 'video_urls']

df['wv3'] = None
df['wv3_g'] = None
df['wv3_e'] = None

df['wv3_TS'] = None
df['wv3_g_TS'] = None
df['wv3_e_TS'] = None

print(df)

# download all audios
count = 1
start = 1
for url in video_url_list[:3]:
    path = YouTube(url).streams.filter(only_audio=True)[0].download(filename=f"audio_{count}.mp3")
    print(f'audio download at : {path}')
    count+=1

print(count)

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
    inp_file = f"./audio_{i+1}.mp3"

    print(f"transcribing {inp_file} using whisper v3 with english and translate")
    result = pipe(inp_file, return_timestamps=True, generate_kwargs={"language": "english", "task": "translate"})
    df.loc[i, 'wv3_e'] =  result["text"]
    df.loc[i, 'wv3_e_TS']  = str(result['chunks'][0])

    print(f"transcribing {inp_file} using whisper v3 with german and translate")
    result = pipe(inp_file, return_timestamps=True, generate_kwargs={"language": "german", "task": "translate"})
    df.loc[i, 'wv3_g'] =  result["text"]
    df.loc[i, 'wv3_g_TS']  = str(result['chunks'][0])

    print(f"transcribing {inp_file} using whisper v3 without keywords")
    result = pipe(inp_file, return_timestamps=True)
    df.loc[i, 'wv3'] =  result["text"]
    df.loc[i, 'wv3_TS']  = str(result['chunks'][0])

    print(df)

    df.to_csv('out.csv')
