# -*- coding: utf-8 -*-
import os
import tempfile
import streamlit as st
import openai
from pydub import AudioSegment
import subprocess # This line is added to fix the NameError

# ----------------------------
# UI Configuration
# ----------------------------
st.set_page_config(layout="wide", page_title="Advanced Subtitle Generator")

st.title("Advanced AI Subtitle Generator (using OpenAI Whisper) 🚀")
st.markdown("""
    Upload a video file. This app uses the **OpenAI Whisper API** for:
    1.  **High Accuracy Transcription:** Get highly accurate text from audio.
    2.  **SRT File Generation:** Create a perfectly timed subtitle file.
""")

# ----------------------------
# Helper Functions
# ----------------------------
def ensure_ffmpeg_access():
    """Ensures ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error("FFmpeg not found. It's required for audio extraction.")
        return False

def extract_audio_pydub(input_path: str, output_path: str) -> str:
    """Uses pydub to extract mono mp3."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio.export(output_path, format="mp3")
        return output_path
    except Exception as e:
        st.error(f"Audio extraction failed: {e}. Check if the file is a valid media format.")
        return None

def transcribe_with_openai(audio_path: str):
    """
    Transcribes audio using OpenAI Whisper API and returns the transcript text.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API Key is not configured in Streamlit Secrets.")
        st.info("Please add your OpenAI API key as a secret named 'OPENAI_API_KEY'.")
        return None

    client = openai.OpenAI(api_key=api_key)
    
    with st.spinner("Uploading audio and transcribing with OpenAI Whisper..."):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="srt"  # Request SRT format directly
                )
        except openai.APIError as e:
            st.error(f"OpenAI API failed: {e}")
            return None
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return None
    
    return transcript

# ----------------------------
# Streamlit App UI
# ----------------------------
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1.  Get your free API key from [OpenAI](https://platform.openai.com/api-keys).
    2.  In your Streamlit app's settings, add a new secret.
    3.  Set the secret name to `OPENAI_API_KEY` and paste your key as the value.
    4.  Upload a video and get your subtitles!
    """
)

uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video_path = os.path.join(tmpdir, uploaded_file.name)
        temp_audio_path = os.path.join(tmpdir, "extracted_audio.mp3")

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if not ensure_ffmpeg_access():
            st.stop()

        st.info("Extracting audio from video...")
        extracted_audio_path = extract_audio_pydub(temp_video_path, temp_audio_path)
        if not extracted_audio_path:
            st.stop()

        srt_content = transcribe_with_openai(extracted_audio_path)

        if srt_content:
            st.success("Transcription Complete!")
            
            st.subheader("SRT Subtitle File")
            st.text_area("SRT Content", srt_content, height=300)
            
            base_name = os.path.splitext(uploaded_file.name)[0]
            dl_name = f"{base_name}_subtitles.srt"
            st.download_button(
                label="Download .SRT File",
                data=srt_content.encode("utf-8"),
                file_name=dl_name,
                mime="text/plain"
            )
        else:
            st.info("No transcript found to generate subtitles.")
