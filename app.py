# -*- coding: utf-8 -*-
import os
import uuid
import math
import tempfile
import subprocess
from datetime import timedelta

import streamlit as st

# Use faster-whisper for word-level timestamps
from faster_whisper import WhisperModel

# Use pydub for audio manipulation (requires ffmpeg)
from pydub import AudioSegment

# ----------------------------
# Configuration - IMPORTANT: Set the path to your ffmpeg executable
# If you downloaded ffmpeg and put it in your root directory, it would be:
FFMPEG_PATH = "./ffmpeg" # Assuming ffmpeg executable is in the root of your repo

# ----------------------------
# Helpers
# ----------------------------
def hhmmss_ms(seconds: float) -> str:
    # SRT time format HH:MM:SS,mmm
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def ensure_ffmpeg_access():
    """
    Checks if the specified FFMPEG_PATH is a valid executable.
    """
    if not os.path.exists(FFMPEG_PATH):
        st.error(f"FFmpeg executable not found at: {FFMPEG_PATH}. Please ensure you have downloaded the correct static build for Linux (e.g., amd64) and placed it in your repository.")
        return False
    
    if not os.access(FFMPEG_PATH, os.X_OK):
        st.error(f"FFmpeg executable at {FFMPEG_PATH} is not executable. Please ensure it has execute permissions (e.g., `chmod +x ffmpeg` locally before pushing to Git).")
        return False
    
    try:
        # Verify it's actually ffmpeg by running its version command
        subprocess.run([FFMPEG_PATH, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        st.success(f"FFmpeg found and accessible at {FFMPEG_PATH}")
        return True
    except subprocess.CalledProcessError:
        st.error(f"FFmpeg at {FFMPEG_PATH} failed to run. It might be corrupted or incompatible.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while verifying FFmpeg: {e}")
        return False

def extract_audio_pydub(input_path: str, output_path: str, sr: int = 16000) -> str:
    """
    Use pydub to extract mono wav @16kHz for transcription stability.
    Uses the specified FFMPEG_PATH directly.
    Returns the path to the extracted audio file.
    """
    try:
        # Set the ffmpeg path for pydub
        AudioSegment.converter = FFMPEG_PATH

        # Load the audio/video file
        audio = AudioSegment.from_file(input_path)
        
        # Set frame rate to 16kHz and channels to mono (1)
        audio = audio.set_frame_rate(sr).set_channels(1)
        
        # Export as WAV (pydub automatically uses the configured ffmpeg for this)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        st.error(f"Error extracting audio with PyDub: {e}. Ensure FFmpeg is accessible at {FFMPEG_PATH} and the input file is valid.")
        return None

# ----------------------------
# Main App
# ----------------------------
st.set_page_config(layout="centered", page_title="Universal Subtitle Translator")
st.title("Universal Subtitle Translator")
st.subheader("Translate with AI!")

st.markdown("""
    This app uses the **Faster-Whisper** model to generate word-level timestamps, 
    and then converts them into a standard SRT subtitle file.
    It supports various audio/video formats by extracting audio using **FFmpeg** (via PyDub).
    
    **Features:**
    - Supports various audio/video formats (mp3, mp4, wav, m4a, flac, mov, avi, mkv).
    - Uses Faster-Whisper for accurate transcription.
    - Generates SRT files with customizable subtitle duration and max characters per line.
    - Automatic language detection or manual hint.
""")

st.markdown("---")

# Model selection
st.sidebar.header("Model Settings")
model_size_options = ["tiny", "base", "small", "medium", "large"]
model_size = st.sidebar.selectbox("Choose Whisper Model Size", model_size_options, index=1)
st.sidebar.info(f"Using {model_size} model. Larger models are more accurate but require more time and memory.")

# Subtitle generation settings
st.sidebar.header("Subtitle Settings")
bucket_seconds = st.sidebar.slider("Max Subtitle Duration (seconds)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
max_chars = st.sidebar.slider("Max Characters per Subtitle Line", min_value=10, max_value=100, value=50, step=5)
lang_hint = st.sidebar.text_input("Language Hint (e.g., 'en' for English, 'auto' for auto-detect)", value="en")

st.markdown("---")

def load_model(model_size="base"):
    home_dir = os.path.expanduser("~")
    model_cache_dir = os.path.join(home_dir, ".cache", "faster-whisper-models")
    
    os.makedirs(model_cache_dir, exist_ok=True)
    
    st.info(f"Using model cache directory: {model_cache_dir}")

    st.info(f"Loading model from cache or downloading to: {model_cache_dir} (This may take a while for larger models)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=model_cache_dir)
    return model

def transcribe_words(audio_path, model, lang=None):
    segments, info = model.transcribe(audio_path, word_timestamps=True, language=lang)
    words = []
    for segment in segments:
        for word in segment.words:
            words.append({"start": word.start, "end": word.end, "word": word.word})
    return words

def bucket_words_by_duration(words, bucket_seconds=3, max_chars_per_sub=50):
    srt_content = []
    current_bucket = []
    current_duration = 0
    sub_idx = 1

    for word_data in words:
        word = word_data["word"]
        word_start = word_data["start"]
        word_end = word_data["end"]

        # Calculate potential text to check character limit
        potential_text = (" ".join([w["word"] for w in current_bucket]) + " " + word).strip() if current_bucket else word

        if not current_bucket:
            bucket_start = word_start
            current_bucket.append(word_data)
            current_duration = word_end - word_start
        else:
            # Check if adding current word exceeds duration or character limit
            if (word_end - bucket_start > bucket_seconds) or (len(potential_text) > max_chars):
                # Close current bucket
                sub_start = hhmmss_ms(bucket_start)
                sub_end = hhmmss_ms(current_bucket[-1]["end"])
                text = " ".join([w["word"] for w in current_bucket]).strip()
                srt_content.append(f"{sub_idx}\n{sub_start} --> {sub_end}\n{text}\n")
                sub_idx += 1

                # Start new bucket with current word
                current_bucket = [word_data]
                bucket_start = word_start
                current_duration = word_end - word_start
            else:
                # Add word to current bucket
                current_bucket.append(word_data)
                current_duration = word_end - bucket_start

    # Add any remaining words in the last bucket
    if current_bucket:
        sub_start = hhmmss_ms(bucket_start)
        sub_end = hhmmss_ms(current_bucket[-1]["end"])
        text = " ".join([w["word"] for w in current_bucket]).strip()
        srt_content.append(f"{sub_idx}\n{sub_start} --> {sub_end}\n{text}\n")

    return "\n".join(srt_content)

# ----------------------------
# Streamlit UI Flow
# ----------------------------
def main():
    temp_audio_path = None
    temp_input_path = None
    temp_dir = None

    # Ensure FFmpeg is available and executable at the start
    st.info("Checking FFmpeg accessibility...")
    if not ensure_ffmpeg_access():
        st.error("FFmpeg setup failed. Please check the logs and ensure the 'ffmpeg' executable is correctly placed and has execute permissions.")
        st.stop() # Stop the app if FFmpeg is not ready

    uploaded = st.file_uploader("Upload Audio/Video", type=["mp3", "mp4", "wav", "m4a", "flac", "mov", "avi", "mkv"])

    try:
        if uploaded:
            st.success("File uploaded successfully!")
            
            temp_dir = tempfile.mkdtemp()
            st.info(f"Temporary directory created at: {temp_dir}")
            
            file_extension = uploaded.name.split('.')[-1].lower()
            temp_input_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{file_extension}")
            
            with open(temp_input_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.info(f"Processing uploaded file: {uploaded.name}")

            # Extract audio to a standard WAV format using PyDub
            st.info("Extracting/Standardizing audio to 16kHz mono WAV using PyDub...")
            with st.spinner("Extracting/Processing audio..."):
                temp_audio_path = extract_audio_pydub(temp_input_path, os.path.join(temp_dir, f"{uuid.uuid4()}.wav"))
                if not temp_audio_path:
                    st.error("Failed to extract or process audio. Please check file format and FFmpeg setup.")
                    return # Stop execution if audio extraction fails
            
            if temp_audio_path: # Proceed only if audio extraction/processing was successful
                with st.spinner(f"Loading Faster-Whisper model: {model_size} ..."):
                    model = load_model(model_size=model_size)

                with st.spinner("Transcribing (word timestamps enabled)... This may take a while."):
                    lang = lang_hint.strip() if lang_hint.strip() != "auto" else None
                    words = transcribe_words(temp_audio_path, model, lang=lang)
                    if not words:
                        st.error("No words were detected. Please try a clearer audio or a different model size.")
                        return

                with st.spinner("Building SRT by fixed duration..."):
                    srt_text = bucket_words_by_duration(words, bucket_seconds=bucket_seconds, max_chars_per_sub=max_chars)

                st.success("Done!")
                st.subheader("Preview (SRT)")
                st.text_area("SRT Content", srt_text, height=320)

                base_name = os.path.splitext(uploaded.name)[0]
                dl_name = f"{base_name}_duration_{int(bucket_seconds*1000)}ms.srt"
                st.download_button(
                    "Download SRT",
                    data=srt_text.encode("utf-8"),
                    file_name=dl_name,
                    mime="text/plain"
                )

        else:
            st.info("Upload an audio or video file to get started!")

    finally:
        # Clean up temporary files and directory
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                st.info(f"Cleaned up temporary audio file: {temp_audio_path}")
            except OSError as e:
                st.warning(f"Could not remove temporary audio file {temp_audio_path}: {e}")
        
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
                st.info(f"Cleaned up temporary input file: {temp_input_path}")
            except OSError as e:
                st.warning(f"Could not remove temporary input file {temp_input_path}: {e}")
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                st.info(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                st.warning(f"Could not remove temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
    main()