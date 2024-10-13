# Audio Transcription App (Whisper and Streamlit)

### Demo
------------------------
![](pics_readme/1013.gif)

### CLICK TO WATCH THE VIDEO
[![Build Your Own Audio Transcription App in Under 10 Minutes!](https://img.youtube.com/vi/MkuttKl5wBk/0.jpg)](https://youtu.be/MkuttKl5wBk)


This repository contains the Python code for an audio transcription web application built using the Hugging Face Whisper model and Streamlit.

- This script utilizes the openai/whisper-small model to transcribe or translate audio files.
- Users can upload audio files and choose the source language (English, Hindi, French) and the task (transcription or translation).
- The app provides transcriptions with timestamps and the option to download the output as a text file.


------

## Installation:

Transcribing mp3 file to numpy requires the FFmpeg. Follow the steps in this link to have it installed. [(How to install FFmpeg in windows)](https://www.wikihow.com/Install-FFmpeg-on-Windows)


You can install these libraries using pip:
```
pip install transformers streamlit
```

To run this app, write this on your command line:
```
streamlit run app.py
```

For low GPU RAM or running it purely on CPU, I recommend you to use the distiled models from WHISPER:
```
import time
from transformers import  pipeline,AutoModelForSpeechSeq2Seq,AutoProcessor
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = "distil-whisper/distil-small.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True
)

model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
    chunk_length_s=15,
    device=device,
)
```

------

## Resources:

-[whisper on hugging face](https://huggingface.co/openai/whisper-small)

-[how to process long audio using whisper](https://huggingface.co/blog/asr-chunking)


