# -*- coding: utf-8 -*-
import os
import uuid
import math
import tempfile
import subprocess
from datetime import timedelta

import streamlit as st

# Use faster-whisper for word-level timestamps
# pip install faster-whisper
from faster_whisper import WhisperModel


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


def ensure_ffmpeg():
    """
    Checks if ffmpeg is available in the system's PATH.
    """
    try:
        # Use shell=True to potentially help with PATH issues in some environments
        result = subprocess.run(["ffmpeg", "-version"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True,
                                shell=True)
        st.success(f"FFmpeg found: {result.stdout.decode().splitlines()[0]}")
        return True
    except subprocess.CalledProcessError as e:
        # If ffmpeg command exists but returns an error
        st.error(f"FFmpeg command failed with error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        # This is the original "ffmpeg not found" error
        st.error("FFmpeg not found in system's PATH. Please ensure it's installed correctly.")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        st.error(f"An unexpected error occurred while checking FFmpeg: {e}")
        return False

def extract_audio_ffmpeg(input_path: str, output_path: str, sr: int = 16000) -> str:
    """
    Use ffmpeg to extract mono wav @16kHz for transcription stability.
    Returns the path to the extracted audio file.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn", # No video
        "-acodec", "pcm_s16le", # PCM 16-bit signed-integer little-endian
        "-ar", str(sr), # Audio sample rate
        "-ac", "1", # Audio channels (1 for mono)
        output_path
    ]
    try:
        # Use shell=True for audio extraction as well
        subprocess.run(cmd, check=True, capture_output=True, shell=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting audio: {e.stderr.decode()}")
        return None # Return None on failure

# ----------------------------
# Main App
# ----------------------------
st.set_page_config(layout="centered", page_title="Universal Subtitle Translator")
st.title("Universal Subtitle Translator")
st.subheader("Translate with AI (Myanmar Language Support!)")

st.markdown("""
    This app uses the **Faster-Whisper** model to generate word-level timestamps, 
    and then converts them into a standard SRT subtitle file.
    It supports multiple languages, including Myanmar.
    
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
lang_hint = st.sidebar.text_input("Language Hint (e.g., 'my' for Burmese, 'en' for English, 'auto' for auto-detect)", value="auto")

st.markdown("---")

def load_model(model_size="base"):
    # Load model from Streamlit's default cache directory (persistent storage)
    # Streamlit Cloud recommends using "~/.cache" for persistent storage
    # Get the home directory first
    home_dir = os.path.expanduser("~")
    model_cache_dir = os.path.join(home_dir, ".cache", "faster-whisper-models")
    
    # Ensure the cache directory exists
    os.makedirs(model_cache_dir, exist_ok=True)
    
    st.info(f"Using model cache directory: {model_cache_dir}")

    # Faster-Whisper's default download_root puts model files directly in it.
    # So we just pass the model_cache_dir as download_root.
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
    current_chars = 0
    sub_idx = 1

    for word_data in words:
        word = word_data["word"]
        word_start = word_data["start"]
        word_end = word_data["end"]

        # Check if adding this word exceeds max duration or max characters
        # If current_bucket is empty, this is the start of a new subtitle
        if not current_bucket:
            bucket_start = word_start
            current_bucket.append(word_data)
            current_duration = word_end - word_start
            current_chars = len(word)
        else:
            potential_duration = word_end - bucket_start
            potential_chars = current_chars + len(word) + (1 if current_bucket else 0) # +1 for space

            # If adding this word exceeds limits, finalize current bucket
            if potential_duration > bucket_seconds or potential_chars > max_chars:
                # Finalize current subtitle
                sub_start = hhmmss_ms(bucket_start)
                sub_end = hhmmss_ms(current_bucket[-1]["end"])
                text = " ".join([w["word"] for w in current_bucket]).strip()
                srt_content.append(f"{sub_idx}\n{sub_start} --> {sub_end}\n{text}\n")
                sub_idx += 1

                # Start new bucket with current word
                current_bucket = [word_data]
                bucket_start = word_start
                current_duration = word_end - word_start
                current_chars = len(word)
            else:
                # Add word to current bucket
                current_bucket.append(word_data)
                current_duration = potential_duration
                current_chars = potential_chars

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
    # Initialize temp_audio_path at the beginning of the function
    temp_audio_path = None 
    temp_input_path = None # Also initialize temp_input_path for cleanup