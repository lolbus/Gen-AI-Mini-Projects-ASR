from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
import torch
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Streamlit app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the whisper model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = pipeline("automatic-speech-recognition", 
                "openai/whisper-large-v3", 
                chunk_length_s=30, 
                stride_length_s=5, 
                return_timestamps=True, 
                device=device)

class TranscriptionRequest(BaseModel):
    language: str
    task: str

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...), 
    language: str = Form(...), 
    task: str = Form(...)
):
    print(f"Language: {language}, Task: {task}")  # Add this line
    try:
        language = language.lower()
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Configure model for language and task
        pipe.model.config.forced_decoder_ids = (
            pipe.tokenizer.get_decoder_prompt_ids(
                language=language,  # Use language directly
                task=task         # Use task directly
            )
        )

        # Perform transcription
        transcription = pipe(temp_file_path, 
                           generate_kwargs={"language": language,  # Use language directly
                                         "task": task})          # Use task directly

        # Format transcription
        formatted_text = ""
        for chunk in transcription['chunks']:
            text = chunk["text"]
            ts = chunk["timestamp"]
            formatted_text += f"[{ts[0]}:{ts[1]}] {text}\n"
        print(f"Final output", formatted_text)

        # Clean up
        os.unlink(temp_file_path)

        return {"success": True, "transcription": formatted_text.strip()}

    except Exception as e:
        return {"success": False, "error": str(e)}

# For Google Colab, add this at the end of your notebook:
'''from pyngrok import ngrok
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)'''

# Create uvicorn config and server
config = uvicorn.Config(app, port=8000, host='0.0.0.0', log_level="info") # Updated to use uvicorn.Config
server = uvicorn.Server(config)

# Start the server in a separate thread
import threading

def start_server():
    server.run()

threading.Thread(target=start_server, daemon=True).start()
!lt --port 8000