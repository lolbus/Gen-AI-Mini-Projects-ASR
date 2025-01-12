import streamlit as st
from transformers import pipeline
import torch
import time
import json
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import os
os.environ['CURL_CA_BUNDLE'] = ''


# Set the page config at the top
st.set_page_config(page_title="Audio-to-Text Transcription", layout="centered", initial_sidebar_state="auto")

# Add API endpoint configuration
api_endpoint = "https://seemingly-ultimate-ape.ngrok-free.app/transcribe"
status_endpoint = "https://seemingly-ultimate-ape.ngrok-free.app/status/"

# Define the app layout
def main():
    st.markdown("<h1 style='color: #00bfff;'>Audio-to-Text Transcription App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>Generate transcription with timestamps and download the result.</p>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
    
    # Select language and task
    languages = ['English', 'Mandarin', 'Malay']  # Choose the source language
    tasks = ['transcribe', 'translate']  # When you choose translate, it translates to English
    
    language = st.selectbox("Choose the language of the audio", options=languages)
    st.write("**When you choose 'translate', it translates the audio to English**.")
    task = st.selectbox("Choose the task", options=tasks)
    
    # Transcribe button
    if uploaded_file is not None:
        if st.button(f"{task.capitalize()}"):
            with st.spinner("Processing..."):
                start_time = time.time()
                
                try:
                    # Prepare the request data
                    files = {'file': uploaded_file}
                    data = {
                        'language': language.lower(),
                        'task': task.lower()
                    }
                    
                    # Send POST request to initiate transcription
                    response = requests.post(
                        api_endpoint,
                        files=files,
                        data=data
                    )
                    loop_count = 0
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            job_id = result.get('job_id')
                            st.success(f"Transcription started! Job ID: {job_id}")
                            
                            # Poll for status
                            with st.spinner("Waiting for transcription to complete..."):
                                while True:
                                    status_response = requests.get(status_endpoint + job_id)
                                    status_data = status_response.json()
                                    
                                    if not status_data.get('success'):
                                        st.error(status_data.get('error', 'Unknown error'))
                                        break
                                    
                                    status = status_data.get('status')
                                    if status == 'completed':
                                        formatted_transcription = status_data.get('result')
                                        st.success(f"{task.capitalize()} completed!")
                                        st.text_area(f"{task.capitalize()} Output", value=formatted_transcription, height=500)
                                        
                                        # Download transcription option
                                        st.download_button("Download Transcription", formatted_transcription, file_name="transcription.txt")
                                        break
                                    elif status == 'failed':
                                        error_message = status_data.get('error', 'Unknown error')
                                        st.error(f"Transcription failed: {error_message}")
                                        break
                                    else:
                                        if loop_count % 20 == 0:
                                            st.info(f"Transcription is still in progress... Current Runtime {time.time() - start_time:.0f}s")
                                        loop_count += 1
                                        time.sleep(3)  # Wait before polling again
                        else:
                            st.error(f"API Error: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"API request failed with status code: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                end_time = time.time()
                st.write(f"Time taken: {round(end_time - start_time, 2)} seconds")

# Helper function to format the transcription with timestamps (not needed anymore as backend handles it)

if __name__ == "__main__":
    main()
