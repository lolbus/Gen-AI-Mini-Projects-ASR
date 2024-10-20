# 🎙️ Live Audio Transcription App using Whisper

This repository contains a Streamlit web app that performs live audio transcription using OpenAI's Whisper model. The app allows users to transcribe or translate audio in real-time using their system's microphone. The transcription is displayed on the web interface with timestamps.

---

## Features

- **Real-time transcription** of audio from your microphone.
- **Language support**: Choose from English, Hindi, or French.
- **Translation mode**: Translate spoken French or Hindi to English.
- **Custom transcription interval**: Control the chunk of audio processed using a slider.
- **GPU Support**: Whisper model is loaded on the GPU (CUDA) for faster inference.
  
---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+**
- **CUDA-supported GPU (optional but recommended)** for faster transcription using Whisper.
- **Git** for cloning the repository.

---

## Installation

### 1. Clone the Repository
```
git clone https://github.com/yourusername/live-transcription-app.git
cd live-transcription-app
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage your dependencies:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate 
```

### 3. Install Dependencies
You can install the required dependencies by running:
```
pip install scipy pyaudiowpatch matplotlib numpy transformers streamlit
```

### 4. Install Additional Dependencies for Windows (Optional)
If you are using Windows and want to use your system's speakers as input, install the pyaudiowpatch library:

``` pip install pyaudiowpatch ```


# Usage
### 1. Running the App
To launch the Streamlit app, run:

``` streamlit run app.py ```

### 2. Setting Up the Microphone and Speakers
Once the app launches, you will be prompted to select your microphone.

Selecting a Loopback Device for Windows:
If you want to transcribe the output from your speakers:

Use the ```pyaudiowpatch``` library to loop back the speaker output to the microphone.

To find the right microphone or speaker device:

```
import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()

print('These are all the available speakers and microphones\n')

for idx in range(p.get_device_count()):
    print(p.get_device_info_by_index(idx))
```
Look for devices where ``` "isLoopbackDevice": True ``` and note the corresponding index.

### 3. Customizing the App
Transcription Interval
You can adjust the transcription interval using the slider in the app, which controls how frequently the app processes audio chunks.

Starting and Stopping Transcription
Use the Start Transcription button to begin recording.
Use the Stop Transcription button to stop the recording process.

----
Example Code Snippets
Below is an example of how the transcription works within the app:

```
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

@st.cache_resource
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to('cuda')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model

# Load the model and processor
processor, model = load_whisper_model()

def transcribe_audio(audio_chunk, processor, model, language, task):
    input_features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to('cuda'), language=language, task=task)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]
```
Speaker and Microphone Selection (for Windows)
To use the system's speakers as the input for transcription:

The pyaudiowpatch library allows loopback devices, but only for Windows systems.
You can use the loopback option to capture audio from speakers and transcribe it using Whisper.

Example Code to List Devices:
```
import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()

for idx in range(p.get_device_count()):
    print(p.get_device_info_by_index(idx))
```
Look for devices with ``` "isLoopbackDevice": True ```.
Select the correct device index and use it for transcription.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Contribution
Feel free to open issues or submit pull requests if you want to improve this project.

# Future Enhancements
- Support for more languages and tasks.
- Improve UI aesthetics.
- Add the ability to save transcriptions in different file formats.
