# @title
from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
import torch
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pyngrok import ngrok
import uuid
import threading
import time
import queue
from ws4py.client.threadedclient import WebSocketClient
import urllib.parse
import ssl
import json
from datetime import timedelta, datetime
import sys
import pyaudio

ssl._create_default_https_context = ssl._create_unverified_context

import contextlib
@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

from datetime import datetime
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


def rate_limited(maxPerSecond):
    minInterval = 1.0 / float(maxPerSecond)
    def decorate(func):
        lastTimeCalled = [time.perf_counter(),0]
        def rate_limited_function(*args,**kargs):
            if lastTimeCalled[1]==0:
                lastTimeCalled[0]=time.perf_counter()
            elapsed = time.perf_counter() - lastTimeCalled[0]
            leftToWait = minInterval*lastTimeCalled[1] - elapsed
            lastTimeCalled[1] += 1
            if leftToWait>0:
                time.sleep(leftToWait)
            ret = func(*args,**kargs)
            return ret
        return rate_limited_function
    return decorate



class AbaxStreamingClient(WebSocketClient):
    def __init__(self, mode, audiofile, url, keywordfile=None, protocols=None, extensions=None, heartbeat_freq=None, byterate=32000,
                 save_adaptation_state_filename=None, ssl_options=None, send_adaptation_state_filename=None):
        # SSL configuration
        self.ssl_options = ssl_options or {}
        self.ssl_options.update({
                "cert_reqs": ssl.CERT_NONE,
                "check_hostname": False,
                "ssl_version": ssl.PROTOCOL_TLS,
            })
        super(AbaxStreamingClient, self).__init__(url, protocols, extensions, heartbeat_freq)        
        self.mode = mode
        self.final_hyps = []
        self.audiofile = audiofile
        self.keywordfile = keywordfile
        self.byterate = byterate
        self.final_hyp_queue = queue.Queue()
        self.save_adaptation_state_filename = save_adaptation_state_filename
        self.send_adaptation_state_filename = send_adaptation_state_filename
        self.processing_complete = False
        self.dt1 = datetime.now()

        '''self.ssl_options = ssl_options or {}

        if self.scheme == "wss":
            # Prevent check_hostname requires server_hostname (ref #187)
            if "cert_reqs" not in self.ssl_options:
                self.ssl_options["cert_reqs"] = ssl.CERT_NONE'''

        self.mode = mode
        with ignoreStderr():
            self.audio = pyaudio.PyAudio()
        self.isStop = False
        self.transcription_idx = 1

    @rate_limited(25)
    def send_data(self, data):
        self.send(data, binary=True)

    def opened(self):
        def send_data_to_ws():
            try:
                if self.mode == 'file':
                    with self.audiofile as audiostream:
                        for block in iter(lambda: audiostream.read(int(self.byterate/25)), ""):
                            self.send_data(block)
                            if len(block) == 0:
                                break
                    self.send("EOS")
            except Exception as e:
                print(f"Error in send_data_to_ws: {e}")
                self.close()

        t = threading.Thread(target=send_data_to_ws)
        t.start()

    def received_message(self, m):
        try:
            response = json.loads(str(m))
            if response['status'] == 0:
                if 'result' in response:
                    trans = response['result']['hypotheses'][0]['transcript']
                    if response['result']['final']:
                        print('generating response...')
                        dt2 = datetime.now()
                        delta = (dt2 - self.dt1).total_seconds()
                        trans = trans.replace("<unk>", "")

                        formatted_text = ""
                        text = trans
                        this_result_start = response['segment-start']
                        this_result_end = response['total-length']
                        srt_start_time = seconds_to_srt_time(this_result_start)
                        srt_end_time = seconds_to_srt_time(this_result_end)
                        formatted_text += f"{self.transcription_idx}\n{srt_start_time} --> {srt_end_time}\n{text}\n\n"
                        self.transcription_idx += 1
                        self.final_hyps.append(formatted_text)
                        print("+" + str(delta) + "ft: " + formatted_text)
            else:
                if 'message' in response:
                    print(f"Server message: {response['message']}")
                    if response['message'] == 'Done':
                        self.processing_complete = True
        except Exception as e:
            print(f"Error processing message: {e}")

    def get_full_hyp(self, timeout=60):
        try:
            # Wait for processing to complete
            start_time = time.time()
            while not self.processing_complete and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self.processing_complete:
                raise TimeoutError("Transcription timed out")

            return "".join(self.final_hyps)
        except Exception as e:
            print(f"Error getting full hypothesis: {e}")
            return ""

    def closed(self, code, reason=None):
        self.processing_complete = True

def seconds_to_srt_time(seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        delta = timedelta(seconds=seconds)
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{int(hours):02}:{int(minutes):02}:{seconds:02},{milliseconds:03}"

# Initialize Ngrok with your auth token
ngrok.set_auth_token("2rWXnt5yL000AS4zUvYTCcRJrwO_25pKFXhtw5kFWw95yLxcF")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Whisper model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-large-v3",
    chunk_length_s=30,
    stride_length_s=5,
    return_timestamps=True,
    device=device
)

# In-memory store for jobs (use a database in production)
jobs = {}

class TranscriptionRequest(BaseModel):
    language: str
    task: str

def process_transcription(job_id: str, file_path: str, language: str, task: str):
    try:
        # Only set forced_decoder_ids for translation task
        if task.lower() == "translate":
            pipe.model.config.forced_decoder_ids = (
                pipe.tokenizer.get_decoder_prompt_ids(
                    language=language,
                    task=task
                )
            )
        else:
            # For transcription, don't set forced_decoder_ids
            pipe.model.config.forced_decoder_ids = None

        # Perform transcription
        transcription = pipe(
            file_path,
            generate_kwargs={
                "language": language,
                "task": task,
            }
        )

        # Format transcription
        formatted_text = ""
        for idx, chunk in enumerate(transcription['chunks'], start=1):
            text = chunk["text"]
            ts = chunk["timestamp"]
            '''
            formatted_text += f"[{ts[0]}:{ts[1]}] {text}\n"'''
            # Dxc format standards
            this_result_start = ts[0]
            this_result_end = ts[1]
            srt_start_time = seconds_to_srt_time(this_result_start)
            srt_end_time = seconds_to_srt_time(this_result_end)
            formatted_text += f"{idx}\n{srt_start_time} --> {srt_end_time}\n{text}\n\n"


        # Store the result
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = formatted_text.strip()

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

    finally:
        # Clean up
        os.unlink(file_path)

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(...),
    task: str = Form(...)
):
    print(f"Language: {language}, Task: {task}")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}

    try:
        language = language.lower()
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Start background thread for transcription
        thread = threading.Thread(
            target=process_transcription,
            args=(job_id, temp_file_path, language, task)
        )
        thread.start()

        return {"success": True, "job_id": job_id}

    except Exception as e:
        return {"success": False, "error": str(e)}

# SpeechLab configuration
ws_url = "wss://gateway.speechlab.sg/client/ws/speech"
api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImR4Y0BzcGVlY2hsYWIuc2ciLCJyb2xlIjoidXNlciIsIm5hbWUiOiJEWEMgVHJpYWwgd2l0aCBNYWxheSBBU1IiLCJ0eXBlIjoibm9ybWFsIiwiaWF0IjoxNzM2NzM1MDc4LCJuYmYiOjE3MzY3MzUwNzgsImV4cCI6MTczOTMyNzA3OCwiaXNzIjoiaHR0cHM6Ly9nYXRld2F5LnNwZWVjaGxhYi5zZyIsInN1YiI6IjY3ODQ3OTE1N2QxNTVlMDQwYTY4ZjZjNiJ9.a_vVRMWygLImyzpEbn5QIN3GNjjrYalWLnUUnlyBJH3PL0fldtE7NFFNFA7jB-1bUhrh6RqfVlh2dG3-QMdovLmKlutcNwqD7Klaewz4IPbHrUvkMl_GpYBPmcS5wrekf2xhXOT_vmkq5StWdkgkj8xf2JPDWfvgc1UES7l8Vz2cSFf_y6pOGziuwQacxuzLaliKvY45Z1W-OmcQkJXu-qp1RsuXJknZjVvnjCbKNlxwtSzzCnP1c9OwCsV6N8vaNwnH5TzkPxs3wY2x3YX8Kty7G6vwQtSa6hxkYK5TOH3vlTbl9Gadg3VocETmK1_Nl7FwAfbVgD9Gv0mImeIhfQ"


class arguments():
    def __init__(self):
        self.mode = 'file'
        self.token = api_key
        self.model = "default"
        self.rate = 32000
        self.save_adaptation_state = None
        self.send_adaptation_state = None
        
        

args = arguments()
def process_sl_transcription(job_id: str, file_path: str, language=None):
    try:
        content_type = ""


        # Create websocket client with the file
        with open(file_path, 'rb') as audiofile:
            # ws = AbaxStreamingClient('file', audiofile, ws_url_with_params)
            ws = AbaxStreamingClient(args.mode, audiofile, 
                    ws_url + '?%s' % (urllib.parse.urlencode([("content-type", content_type)])) + '&%s' % (urllib.parse.urlencode([("accessToken", args.token)])) + '&%s' % (urllib.parse.urlencode([("token", args.token)])) + '&%s' % (urllib.parse.urlencode([("model", args.model)])), 
                    byterate=args.rate,
                    save_adaptation_state_filename=args.save_adaptation_state, send_adaptation_state_filename=args.send_adaptation_state)
            try:
                ws.connect()
                # Wait for the complete transcription
                result = ws.get_full_hyp(timeout=300)  # 5 minutes timeout

                
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['result'] = result.strip()
                

            except TimeoutError:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = "Transcription timed out"
            except Exception as e:
                print(f"WebSocket error: {e}")
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = str(e)
            finally:
                try:
                    ws.close()
                except:
                    pass

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
    finally:
        try:
            os.unlink(file_path)
        except:
            pass

@app.post("/transcribe-sl")
async def transcribe_audio_sl(
    file: UploadFile = File(...),
    language: str = Form(...),
    task: str = Form(...)
):
    print(f"Language: {language}, Task: {task}")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Start background thread for transcription
        thread = threading.Thread(
            target=process_sl_transcription,
            args=(job_id, temp_file_path)
        )
        thread.start()

        return {"success": True, "job_id": job_id}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"success": False, "error": "Job ID not found."}
    return {
        "success": True,
        "status": job['status'],
        "result": job.get('result'),
        "error": job.get('error')
    }

# Expose the API via Ngrok
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()


    # Expose port 8000 via ngrok
    public_url = ngrok.connect(8000, domain="seemingly-ultimate-ape.ngrok-free.app")
    print("Ngrok URL:", public_url)
    start_time = time.time()
    sec = 0
    while True:
        sec = int(time.time() - start_time)
        print(f"Script running runtime: {sec}s")
        time.sleep(600)



