# -*- coding: utf-8 -*-
import os
import tempfile
import subprocess
import streamlit as st
import assemblyai as aai
from pydub import AudioSegment

# ----------------------------
# UI Configuration
# ----------------------------
st.set_page_config(layout="wide", page_title="Advanced Subtitle Generator")

st.title("Advanced AI Subtitle Generator (using AssemblyAI) 🚀")
st.markdown("""
    Upload a video file. This app uses the **AssemblyAI API** for:
    1.  **Best Accuracy Transcription:** Uses AssemblyAI's best model for top-tier accuracy.
    2.  **Speaker Labels:** Automatically detect and label different speakers.
    3.  **Automatic Summarization:** Generate a summary of the conversation.
    4.  **Perfectly Formatted SRT:** Create a subtitle file with accurate timing and punctuation.
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
    """Uses pydub to extract mono wav @16kHz."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        st.error(f"Audio extraction failed: {e}. Check if the file is a valid media format.")
        return None

def transcribe_with_assemblyai(audio_path: str):
    """
    Transcribes audio using AssemblyAI API and returns the transcript object.
    """
    api_key = st.secrets.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        st.error("AssemblyAI API Key is not configured in Streamlit Secrets.")
        st.info("Please add your AssemblyAI API key as a secret named 'ASSEMBLYAI_API_KEY'.")
        return None

    aai.settings.api_key = api_key
    
    # Enhanced configuration for best accuracy and formatting
    config = aai.TranscriptionConfig(
    speech_model=aai.SpeechModel.best,
    punctuate=True,
    format_text=True,
    speaker_labels=True,
    auto_highlights=True,
    extract_text=True
    )

    transcriber = aai.Transcriber()
    
    with st.spinner("Uploading audio and transcribing with AssemblyAI... This is usually very fast!"):
        try:
            transcript = transcriber.transcribe(audio_path, config)
        except Exception as e:
            st.error(f"AssemblyAI transcription failed: {e}")
            return None

    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Transcription failed: {transcript.error}")
        return None
    
    return transcript

# ----------------------------
# Streamlit App UI
# ----------------------------
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1.  Get your free API key from [AssemblyAI](https://www.assemblyai.com/).
    2.  In your Streamlit app's settings, add a new secret.
    3.  Set the secret name to `ASSEMBLYAI_API_KEY` and paste your key as the value.
    4.  Upload a video and enjoy the advanced features!
    """
)

uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video_path = os.path.join(tmpdir, uploaded_file.name)
        temp_audio_path = os.path.join(tmpdir, "extracted_audio.wav")

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if not ensure_ffmpeg_access():
            st.stop()

        st.info("Extracting audio from video...")
        extracted_audio_path = extract_audio_pydub(temp_video_path, temp_audio_path)
        if not extracted_audio_path:
            st.stop()

        transcript = transcribe_with_assemblyai(extracted_audio_path)

        if transcript:
            st.success("Transcription Complete!")
            
            # Display results in tabs
            tabs = ["📄 SRT Subtitles", "📝 Full Transcript", "👥 Speakers", "💡 Summary"]
    if transcript.text_extractions:
        tabs.append("🔤 On-Screen Text (OCR)")

    all_tabs = st.tabs(tabs)
            with tab1:
                st.subheader("SRT Subtitle File")
                if transcript.text:
                    srt_content = transcript.export_subtitles_srt()
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

            with tab2:
                st.subheader("Full Transcript Text")
                st.text_area("Full Text", transcript.text, height=300)

            with tab3:
                st.subheader("Transcript by Speaker")
                if transcript.utterances:
                    for utterance in transcript.utterances:
                        st.markdown(f"**Speaker {utterance.speaker}:** {utterance.text}")
                else:
                    st.info("No speaker labels were detected in this audio.")

            with tab4:
                st.subheader("Conversation Summary")
                if transcript.highlights:
                    for result in transcript.highlights.results:
                        st.markdown(f"- {result.text}")
                else:
                    st.info("No summary could be generated for this audio.")

    if transcript.text_extractions:
        with all_tabs[-1]: # The last tab
            st.subheader("Detected On-Screen Text (OCR)")
            for ocr_result in transcript.text_extractions:
                st.markdown(f"**Timestamp: `{ocr_result.timestamp.start}` - `{ocr_result.timestamp.end}`**")
                st.info(ocr_result.text)
