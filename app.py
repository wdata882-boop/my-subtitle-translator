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
    3.  **On-Screen Text Recognition (OCR):** Extracts text visible in the video.
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
    
    # Enhanced configuration with OCR feature enabled
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        punctuate=True,
        format_text=True,
        speaker_labels=True,
        auto_highlights=True,
        extract_text=True  # OCR feature is now enabled
    )

    transcriber = aai.Transcriber()
    
    with st.spinner("Uploading file and processing with AssemblyAI... This may take a moment."):
        try:
            # For video files with OCR, we submit the video path directly
            transcript = transcriber.transcribe(audio_path, config)
        except Exception as e:
            st.error(f"AssemblyAI processing failed: {e}")
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

# NOTE: Since we are using OCR, we will upload the video file directly to AssemblyAI.
# We no longer need to extract the audio first.
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the uploaded video file to a temporary path
        temp_video_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # We pass the VIDEO file path directly to the transcription function for OCR
        transcript = transcribe_with_assemblyai(temp_video_path)

        if transcript:
            st.success("Processing Complete!")
            
            # Dynamically create tabs based on available results
            tab_titles = ["📄 SRT Subtitles", "📝 Full Transcript", "👥 Speakers", "💡 Summary"]
            if transcript.text_extractions:
                tab_titles.append("🔤 On-Screen Text (OCR)")
            
            tabs = st.tabs(tab_titles)

            # Tab 1: SRT Subtitles
            with tabs[0]:
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
                    st.info("No spoken words found to generate subtitles.")

            # Tab 2: Full Transcript
            with tabs[1]:
                st.subheader("Full Transcript Text")
                st.text_area("Full Text", transcript.text, height=300)

            # Tab 3: Speakers
            with tabs[2]:
                st.subheader("Transcript by Speaker")
                if transcript.utterances:
                    for utterance in transcript.utterances:
                        st.markdown(f"**Speaker {utterance.speaker}:** {utterance.text}")
                else:
                    st.info("No speaker labels were detected in this audio.")

            # Tab 4: Summary
            with tabs[3]:
                st.subheader("Conversation Summary")
                if transcript.highlights:
                    for result in transcript.highlights.results:
                        st.markdown(f"- {result.text}")
                else:
                    st.info("No summary could be generated for this audio.")
            
            # Tab 5: OCR Results (if any)
            if transcript.text_extractions:
                with tabs[4]:
                    st.subheader("Detected On-Screen Text (OCR)")
                    for ocr_result in transcript.text_extractions:
                        # Format timestamp for better readability
                        start_ms = ocr_result.timestamp.start
                        end_ms = ocr_result.timestamp.end
                        st.markdown(f"**Time:** `{start_ms//1000}s` to `{end_ms//1000}s`")
                        st.info(ocr_result.text)
