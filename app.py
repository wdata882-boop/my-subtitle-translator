# -*- coding: utf-8 -*-
import os
import math
import tempfile
import subprocess
import streamlit as st
from faster_whisper import WhisperModel
from pydub import AudioSegment

# ----------------------------
# Constants and UI Configuration
# ----------------------------
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL_SIZE = "base"
MAX_CHARS_PER_SUB = 60
DEFAULT_BUCKET_SECONDS = 5
DEFAULT_MIN_SUBTITLE_DURATION = 0.5

st.set_page_config(layout="wide", page_title="Universal Subtitle Generator")
st.title("Universal Subtitle Generator 🎬")
st.markdown("""
    Upload a video file to generate a timed SRT subtitle file.
    This app uses **Faster-Whisper** for fast and accurate transcription, with word-level timestamps, **without needing an API key**.
""")

# ----------------------------
# Helper Functions
# ----------------------------
@st.cache_resource
def load_model(model_size: str):
    """
    Loads the Faster-Whisper model from cache.
    """
    return WhisperModel(model_size, device="cpu")

def ensure_ffmpeg_access():
    """Ensures ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error("FFmpeg not found. It's required for audio extraction.")
        return False

def hhmmss_ms(seconds: float) -> str:
    """
    Converts seconds to SRT time format HH:MM:SS,mmm
    """
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def transcribe_words(_model: WhisperModel, audio_path: str):
    """
    Transcribes the audio and returns a list of words with their start/end times.
    """
    try:
        segments, info = _model.transcribe(audio_path, word_timestamps=True)
        words = []
        for segment in segments:
            for word in segment.words:
                words.append(word)
        return words
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

def bucket_words_by_duration(words, bucket_seconds: int = 5, max_chars_per_sub: int = 60, min_subtitle_duration: float = 0.5):
    """
    Generates SRT content by grouping words into subtitle lines based on duration and max characters.
    """
    srt_content = ""
    current_subtitle = ""
    start_time = 0.0
    subtitle_index = 1
    
    if not words:
        return ""

    for i, word in enumerate(words):
        if not current_subtitle:
            start_time = word.start
        
        next_word_text = word.word.strip()
        current_subtitle_plus_next = current_subtitle + " " + next_word_text if current_subtitle else next_word_text

        if len(current_subtitle_plus_next) > max_chars_per_sub or \
           (word.end - start_time) > bucket_seconds or \
           i == len(words) - 1:
            
            end_time = word.end
            
            if (end_time - start_time) < min_subtitle_duration:
                end_time = start_time + min_subtitle_duration

            srt_content += f"{subtitle_index}\n"
            srt_content += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(end_time)}\n"
            srt_content += f"{current_subtitle.strip()}\n\n"
            
            current_subtitle = next_word_text
            start_time = word.start
            subtitle_index += 1
            
        else:
            current_subtitle = current_subtitle_plus_next
    
    if current_subtitle:
        last_word = words[-1]
        start_time_last = words[len(words) - len(current_subtitle.split())].start
        end_time_last = last_word.end
        
        if (end_time_last - start_time_last) < min_subtitle_duration:
             end_time_last = start_time_last + min_subtitle_duration
        
        srt_content += f"{subtitle_index}\n"
        srt_content += f"{hhmmss_ms(start_time_last)} --> {hhmmss_ms(end_time_last)}\n"
        srt_content += f"{current_subtitle.strip()}\n\n"

    return srt_content

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

def main():
    """Main function for the Streamlit app."""
    st.sidebar.header("Settings")
    model_size = st.sidebar.selectbox("Choose Whisper Model Size", MODEL_SIZES, index=MODEL_SIZES.index(DEFAULT_MODEL_SIZE))
    bucket_seconds = st.sidebar.slider("Max Subtitle Duration (seconds)", min_value=1, max_value=15, value=DEFAULT_BUCKET_SECONDS, step=1)
    max_chars = st.sidebar.slider("Max Characters per Subtitle", min_value=20, max_value=120, value=MAX_CHARS_PER_SUB, step=5)
    min_duration_seconds = st.sidebar.slider("Min Subtitle Duration (seconds)", min_value=0.1, max_value=2.0, value=DEFAULT_MIN_SUBTITLE_DURATION, step=0.1)

    uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Generate Subtitles"):
            if not ensure_ffmpeg_access():
                st.stop()

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_video_path = os.path.join(tmpdir, uploaded_file.name)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                temp_audio_path = os.path.join(tmpdir, "extracted_audio.wav")
                st.info("Extracting audio from video...")
                extracted_audio_path = extract_audio_pydub(temp_video_path, temp_audio_path)
                
                if extracted_audio_path is None: 
                    st.stop()

                with st.spinner(f"Loading Faster-Whisper model: {model_size} ..."):
                    model = load_model(model_size=model_size)

                with st.spinner("Transcribing (word timestamps enabled)... This may take a while depending on audio length and model size."):
                    words = transcribe_words(_model=model, audio_path=extracted_audio_path) 
                    if not words:
                        st.error("No words were detected. Please try a clearer audio or a different model size.")
                        st.stop()

                with st.spinner("Building SRT with intelligent segmentation..."):
                    srt_text = bucket_words_by_duration(
                        words, 
                        bucket_seconds=bucket_seconds, 
                        max_chars_per_sub=max_chars, 
                        min_subtitle_duration=min_duration_seconds
                    )

                st.success("Done!")
                st.subheader("Preview (SRT)")
                st.text_area("SRT Content", srt_text, height=320)

                base_name = os.path.splitext(uploaded_file.name)[0]
                dl_name = f"{base_name}_sub.srt"
                
                st.download_button(
                    "Download SRT File",
                    srt_text.encode("utf-8"),
                    file_name=dl_name,
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
