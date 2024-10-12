# pip install PyAudioWPatch

from scipy.signal import resample
import pyaudiowpatch as pyaudio
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import wave
import soundcard as sc
import soundfile as sf

def save_wave(audio_frames, idx, CHANNELS, FORMAT, p, WISPER_RATE):
    # Save the recorded audio to a WAV file
    wav_file = f"temp_audio_{idx}.wav"
    with wave.open(wav_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(WISPER_RATE)
        wf.writeframes(b''.join(audio_frames))




MIXED_PRECISION = True
INPUT_DEVICE = {1:'pc_micro_phone', 2:'pc_speaker', 3:'bluetooth_speaker', 4:'bluetooth_microphone'}
""" for pc_speaker, go to the sound icon and right click-> choose sounds -> recordings -> choose streo Mix as default  """
INPUT_DEVICE_IDX = 3


device='cuda'
# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
# forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")#for french to french
# forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="translate")#for french to english
# forced_decoder_ids = None # for english to english
forced_decoder_ids = processor.get_decoder_prompt_ids(language="hindi", task="translate")#for french to english


# Initialize PyAudio
p = pyaudio.PyAudio()

#  TODO create a nice function from this
if INPUT_DEVICE[INPUT_DEVICE_IDX] == 'pc_speaker':
    default_speakers = p.get_default_input_device_info()

elif INPUT_DEVICE[INPUT_DEVICE_IDX] == 'bluetooth_speaker':
    # Get default WASAPI info
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    # Get default WASAPI speakers
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    if not default_speakers["isLoopbackDevice"]:
        for loopback in p.get_loopback_device_info_generator():
            """
            Try to find loopback device with same name(and [Loopback suffix]).
            Unfortunately, this is the most adequate way at the moment.
            """
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                break
else:
    pass

print(f'The loopback device is {default_speakers}')

# Settings for recording audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = int(default_speakers['defaultSampleRate'] ) 
WISPER_RATE = 16000# Whisper expects 16kHz input
CHUNK = 1024  # Number of frames per buffer
TRANSCRIPTION_INTERVAL = 30  # Interval for transcription in seconds

# Open a stream to record audio

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=default_speakers["index"])

print("Listening for audio... Speak now.")

audio_buffer = np.array([], dtype=np.float32)  # Buffer to store accumulated audio
last_transcription_time = time.time()  # Initialize the last transcription time
audio_frames = []  # Store raw audio frames

# Open a text file to save the transcriptions
transcription_file = open("transcriptions_cpu.txt", "a")


while True:
    try:

        # Read a chunk of audio
        data = stream.read(CHUNK, exception_on_overflow=False)  # Read a chunk of audio (1024 samples per chunk
        audio_frames.append(data)  # Save raw audio data for MP3 conversion
        audio_chunk = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        downsampled_chunk = resample(audio_chunk, int(len(audio_chunk) * WISPER_RATE / RATE))  # Resample to 16kHz

        # print(np.abs(audio_chunk).mean()> 0.02)

        # Only append audio with sound (ignoring silence)
        if np.abs(audio_chunk).mean() > 0.01:
            audio_buffer = np.append(audio_buffer, downsampled_chunk)

        # Check if it's time to perform transcription
        current_time = time.time()
        if current_time - last_transcription_time >= TRANSCRIPTION_INTERVAL:
            if audio_buffer.size > 0:  # Ensure there's audio to transcribe

                start_translation_time = time.time()

                input_features = processor(audio_buffer, sampling_rate=WISPER_RATE, return_tensors="pt").input_features
                # Generate token ids
                predicted_ids = model.generate(input_features.to(device), forced_decoder_ids=forced_decoder_ids)
                # Decode token ids to text
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                transcription_text = transcription[0]

                translation_duration = time.time() - start_translation_time

                # Save the transcription with a timestamp to the file
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                transcription_file.write(f"[{timestamp}] {transcription_text}\n")
                transcription_file.flush()  # Ensure it's written to the file immediately


                # Print the transcription
                print(f"Transcription ({translation_duration:0.3f}s): {transcription_text}")

                # Clear buffer after transcription
                audio_buffer = np.array([], dtype=np.float32)

            last_transcription_time = current_time  # Update last transcription time

    except KeyboardInterrupt:
        print("Stopped listening.")
        save_wave(audio_frames, idx=1, CHANNELS=CHANNELS, FORMAT=FORMAT, p=p, WISPER_RATE=WISPER_RATE)
        break

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()




