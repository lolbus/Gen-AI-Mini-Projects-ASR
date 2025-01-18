import streamlit as st
import time
import json
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import os

os.environ['CURL_CA_BUNDLE'] = ''

#Updates to do: Add a button to download the audio from youtube and transcribe it, 
# Auto convert mp4/mp3 uploads to .wav
# Add the English_Malay_v4 SL selection, 
# develop transcription autonaming 

# Set the page config at the top
st.set_page_config(page_title="Audio-to-Text Transcription", layout="centered", initial_sidebar_state="auto")

# Add API endpoint configuration
api_endpoint_whisper = "https://seemingly-ultimate-ape.ngrok-free.app/transcribe"
api_endpoint_sl = "https://seemingly-ultimate-ape.ngrok-free.app/transcribe-sl"
status_endpoint = "https://seemingly-ultimate-ape.ngrok-free.app/status/"



# Define the app layout
def main():
    # Initialize session state using st.session_state.setdefault instead
    st.session_state.setdefault('formatted_transcription', None)
    st.session_state.setdefault('transcription_file_name', '')
    
    st.markdown("<h1 style='color: #00bfff;'>Audio-to-Text Transcription App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>Generate transcription with timestamps and download the result.</p>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file. (Speechlab service only can handle wav)", type=["wav", "mp3"])
    transcription_file_name = ""
    if uploaded_file is not None:
        st.audio(uploaded_file)
        transcription_file_name = uploaded_file.name.replace(".mp3", "").replace(".wav", "").replace(".ogg", "")
    
    # Select language and task
    languages = ['English', 'Mandarin', 'Malay']  # Choose the source language
    tasks = ['transcribe', 'translate']  # When you choose translate, it translates to English
    models = ['whisper', 'default-speechlab', 'english_malay_v4-speechlab']
    
    language = st.selectbox("Choose the language of the audio", options=languages)
    model = st.selectbox("Choose the model", options=models)
    transcription_file_name += "_" + model + ".txt"
    if model == 'whisper':
        api_endpoint = api_endpoint_whisper
    else:
        api_endpoint = api_endpoint_sl
    task = st.selectbox("Choose the task", options=tasks)
    st.write("**Note: When you choose 'translate' task, it translates the audio to English Text**.")
    
    # Transcribe button
    if uploaded_file is not None:
        if st.button(f"{task.capitalize()}"):
            # with st.spinner("Processing..."):
            st.info("Processing... Initiating a Transcription Job...")
            start_time = time.time()
            try:
                # Prepare the request data
                files = {'file': uploaded_file}
                data = {
                    'language': language.lower(),
                    'task': task.lower(),
                    'model': model.lower()
                }
                
                # Send POST request to initiate transcription
                response = requests.post(
                    api_endpoint,
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        job_id = result.get('job_id')
                        st.success(f"Transcription started! Job ID: {job_id}")
                        
                        # Poll for status
                        # with st.spinner("Waiting for transcription to complete..."):
                        progress_placeholder = st.empty()  # Add this line here
                        while True:
                            status_response = requests.get(status_endpoint + job_id, timeout=300)
                            status_data = status_response.json()
                            
                            if not status_data.get('success'):
                                st.error(status_data.get('error', 'Unknown error'))
                                break
                            
                            status = status_data.get('status')
                            if status == 'completed':
                                st.session_state.formatted_transcription = status_data.get('result')
                                st.success(f"{task.capitalize()} completed!")
                                break
                            elif status == 'failed':
                                error_message = status_data.get('error', 'Unknown error')
                                st.error(f"Transcription failed: {error_message}")
                                break
                            else:
                                progress_placeholder.info(f"Transcription is still in progress... Current Runtime: {time.time() - start_time:.0f}s. Inference status: {status}")
                                time.sleep(5)  # Wait before polling again
                    else:
                        st.error(f"API Error: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"API request failed with status code: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
            
            end_time = time.time()
            st.write(f"Time taken: {round(end_time - start_time, 2)} seconds")
    
    # Display transcription only once, outside the processing block
    if st.session_state.formatted_transcription is not None:
        st.text_area(
            "Transcription Output", 
            value=st.session_state.formatted_transcription, 
            height=500,
            key="transcription_output"
        )
        if transcription_file_name:  # Only show download button if we have a filename
            st.download_button(
                "Download Transcription", 
                st.session_state.formatted_transcription, 
                file_name=transcription_file_name,
                key="download_button"
            )
    

# Helper function to format the transcription with timestamps (not needed anymore as backend handles it)

if __name__ == "__main__":
    main()
