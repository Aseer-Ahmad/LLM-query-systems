from transformers import Wav2Vec2Processor, Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import librosa
import nltk

def load_data(input_file):
    #reading the file
    speech, sample_rate = librosa.load(input_file)
    #make it 1-D
    if len(speech.shape) > 1: 
        speech = speech[:,0] + speech[:,1]
    #Resampling the audio at 16KHz
    if sample_rate !=16000:
        speech = librosa.resample(speech, orig_sr = sample_rate,target_sr = 16000)
    
    return speech

def correct_casing(input_sentence):

  sentences = nltk.sent_tokenize(input_sentence)
  return (' '.join([s.replace(s[0],s[0].capitalize(),1) for s in sentences]))

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
# load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# tokenize
# input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1
inp_file = 'audio.mp3'
speech = load_data(inp_file)
input_values = tokenizer(speech, return_tensors="pt").input_values

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids)

#Correcting the letter casing
transcription = correct_casing(transcription.lower())
print(transcription)