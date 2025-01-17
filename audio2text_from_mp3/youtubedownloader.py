

from pytubefix import YouTube
import ffmpeg

# Replace with your YouTube video URL
video_url = 'https://www.youtube.com/watch?v=lhmKCkZusMo'

try:
    # Create a YouTube object with additional configuration
    yt = YouTube(
        video_url,
        use_oauth=True,
        allow_oauth_cache=True
    )
    
    
    # Get the lowest resolution stream
    video_stream = yt.streams.get_lowest_resolution()
    
    safe_filename = "".join([c for c in yt.title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    
    # Define file paths
    mp4_path = f"{safe_filename}.mp4"
    
    # Download the video
    print(f"Downloading: {yt.title}")
    video_stream.download(output_path=".", filename=mp4_path)
    print("Download complete!")

    wav_path = f"{safe_filename}.wav"

    # Convert to WAV using ffmpeg-python
    print("Converting to WAV...")
    stream = ffmpeg.input(mp4_path)
    stream = ffmpeg.output(
        stream,
        wav_path,
        acodec='pcm_s16le',  # 16-bit PCM
        ac=1,                # mono audio
        ar=16000,           # 16kHz sampling rate
        vn=None             # no video
    )
    ffmpeg.run(stream, overwrite_output=True)
    print("Conversion complete!")
    
    # Optionally, remove the MP4 file
    # os.remove(mp4_path)
    print(f"Created WAV file: {wav_path}")


    
    

except Exception as e:
    print(f"An error occurred: {e}")