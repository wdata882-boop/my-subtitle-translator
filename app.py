# -*- coding: utf-8 -*-
import os
import math
import tempfile
import subprocess

import streamlit as st

# Use faster-whisper for word-level timestamps
# pip install faster-whisper
from faster_whisper import WhisperModel

# Use pydub for audio extraction with explicit ffmpeg path
# pip install pydub
from pydub import AudioSegment

# ----------------------------
# Constants and UI Configuration
# ----------------------------
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL_SIZE = "base"
MAX_CHARS_PER_SUB = 60 # Maximum characters per subtitle line
DEFAULT_BUCKET_SECONDS = 5 # Default duration for subtitle buckets

# FFmpeg will be installed via packages.txt and should be in the system's PATH
# No need to specify a local path like "./ffmpeg"
# FFMPEG_PATH = "./ffmpeg" # This line is removed / commented out

# ----------------------------
# Helper Functions
# ----------------------------
def hhmmss_ms(seconds: float) -> str:
    """
    Converts seconds to SRT time format HH:MM:SS,mmm
    """
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

@st.cache_resource
def load_model(model_size: str):
    """
    Loads the Faster-Whisper model. Uses cache for efficiency.
    """
    st.info(f"Loading Faster-Whisper model: {model_size}. This may take a while for larger models.")
    # For Streamlit Cloud, setting device to "cpu" is generally safer and sufficient.
    # compute_type can be "int8" for CPU for better performance if supported.
    # CHANGED: Reverted compute_type to "int8" because "int8_float16" is not supported on Streamlit Cloud's CPUs.
    return WhisperModel(model_size, device="cpu", compute_type="int8") 

@st.cache_resource
def transcribe_words(_model: WhisperModel, audio_path: str, lang: str = None):
    """
    Transcribes audio using Faster-Whisper with word-level timestamps.
    """
    st.info("Transcribing audio with word-level timestamps...")
    segments, _ = _model.transcribe(
        audio_path,
        word_timestamps=True,
        language=lang, 
        # CHANGED: Increased beam_size for potentially better precision.
        # This will make transcription slower, but can improve accuracy of timestamps.
        beam_size=7 
    )
    words = []
    for segment in segments:
        for word in segment.words:
            words.append(word)
    return words

def bucket_words_by_duration(words: list, bucket_seconds: int = 5, max_chars_per_sub: int = 60) -> str:
    """
    Creates SRT content by bucketing words into fixed duration intervals.
    Attempts to break lines by max_chars_per_sub.
    """
    srt_content = []
    subtitle_idx = 1
    current_bucket_start_time = 0.0
    current_words_in_bucket = []

    for i, word in enumerate(words):
        # If the word's start time exceeds the current bucket's end time,
        # or if we have accumulated too many characters,
        # or if it's the last word, finalize the current bucket.
        combined_text_length = len(" ".join([w.word for w in current_words_in_bucket] + [word.word]))

        if word.start >= current_bucket_start_time + bucket_seconds or \
           (combined_text_length > max_chars_per_sub and current_words_in_bucket) or \
           i == len(words) - 1:
            
            if current_words_in_bucket:
                # Add the current bucket as a subtitle entry
                subtitle_text = " ".join([w.word.strip() for w in current_words_in_bucket]).strip()
                
                # If the current subtitle is too long, try to break it by character limit
                if len(subtitle_text) > max_chars_per_sub:
                    # Find a good breaking point (e.g., after a comma or space)
                    break_point = subtitle_text.rfind(' ', 0, max_chars_per_sub)
                    if break_point == -1: # No space found within limit, just break
                        break_point = max_chars_per_sub
                    
                    line1 = subtitle_text[:break_point].strip()
                    line2 = subtitle_text[break_point:].strip()
                    subtitle_text = f"{line1}\n{line2}" # Add newline for 2nd line
                
                start_time_str = hhmmss_ms(current_words_in_bucket[0].start)
                end_time_str = hhmmss_ms(current_words_in_bucket[-1].end)
                
                srt_content.append(f"{subtitle_idx}")
                srt_content.append(f"{start_time_str} --> {end_time_str}")
                srt_content.append(subtitle_text)
                srt_content.append("") # Empty line for SRT format
                subtitle_idx += 1

            # Start a new bucket with the current word
            # Ensure new bucket start time is correctly aligned
            if i == len(words) - 1 and not current_words_in_bucket: # Handle case where last word forms a new bucket alone
                current_bucket_start_time = word.start
            elif current_words_in_bucket: # If there were words, next bucket starts after previous one finished
                current_bucket_start_time = math.floor(current_words_in_bucket[-1].end / bucket_seconds) * bucket_seconds
            else: # Fallback for initial state or empty bucket
                 current_bucket_start_time = math.floor(word.start / bucket_seconds) * bucket_seconds

            current_words_in_bucket = [word]
        else:
            current_words_in_bucket.append(word)

    # Handle the very last word/bucket if it was just added to current_words_in_bucket
    # and didn't trigger the 'if' condition in the last iteration, or was the only word.
    if current_words_in_bucket and (not srt_content or (srt_content and srt_content[-1] != "")): # check if last entry was a subtitle
        subtitle_text = " ".join([w.word.strip() for w in current_words_in_bucket]).strip()
        if len(subtitle_text) > max_chars_per_sub:
            break_point = subtitle_text.rfind(' ', 0, max_chars_per_sub)
            if break_point == -1:
                break_point = max_chars_per_sub
            line1 = subtitle_text[:break_point].strip()
            line2 = subtitle_text[break_point:].strip()
            subtitle_text = f"{line1}\n{line2}"
        
        start_time_str = hhmmss_ms(current_words_in_bucket[0].start)
        end_time_str = hhmmss_ms(current_words_in_bucket[-1].end)
        
        srt_content.append(f"{subtitle_idx}")
        srt_content.append(f"{start_time_str} --> {end_time_str}")
        srt_content.append(subtitle_text)
        srt_content.append("")

    return "\n".join(srt_content)

def ensure_ffmpeg_access():
    """
    Ensures the ffmpeg executable is available in the system PATH.
    It will be installed via packages.txt during deployment.
    """
    try:
        # Use "ffmpeg" directly as it will be in the system PATH after packages.txt installation
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except FileNotFoundError:
        st.error(f"FFmpeg executable not found. It should be installed via 'packages.txt'. Please check your packages.txt file and ensure it contains 'ffmpeg'.")
        return False
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg returned an error during version check: {e}. This might indicate a problem with the ffmpeg installation via packages.txt.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during FFmpeg setup: {e}. Please check your deployment logs for more details.")
        return False

def extract_audio_pydub(input_path: str, output_path: str, sr: int = 16000) -> str:
    """
    Use pydub to extract mono wav @16kHz.
    Pydub will automatically find 'ffmpeg' in the system PATH.
    Returns the path to the extracted audio if successful, None otherwise.
    """
    try:
        # Set pydub's converter to "ffmpeg" as it's in system PATH now
        AudioSegment.converter = "ffmpeg"
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(sr)  # 16kHz
        audio.export(output_path, format="wav")
        st.success(f"Audio extracted successfully to {output_path}")
        return output_path
    except Exception as e:
        st.error(f"Error extracting audio with pydub (FFmpeg issue?): {e}. Please check your video file format and ensure 'ffmpeg' is installed correctly via packages.txt.")
        # Important: Return None if extraction fails, so the calling code knows it failed
        return None

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(layout="wide", page_title="Universal Subtitle Generator")

st.title("Universal Subtitle Generator 🎬")
st.markdown("""
    Upload a video file, and this app will:
    1. Extract audio using FFmpeg (via pydub).
    2. Transcribe the audio into **English** using the Faster-Whisper model with word-level timestamps.
    3. Generate an SRT subtitle file, breaking lines into configurable duration buckets and character limits.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_size = st.sidebar.selectbox("Choose Whisper Model Size", MODEL_SIZES, index=MODEL_SIZES.index(DEFAULT_MODEL_SIZE))
# CHANGED: Removed language hint input if always generating English subtitles
# lang_hint = st.sidebar.text_input("Source Language Hint (e.g., 'en', 'my', 'zh', 'ja' - optional)", value="")
bucket_seconds = st.sidebar.slider("Subtitle Duration Bucket (seconds)", min_value=1, max_value=10, value=DEFAULT_BUCKET_SECONDS)
max_chars = st.sidebar.slider("Max Characters Per Subtitle Line", min_value=20, max_value=100, value=MAX_CHARS_PER_SUB)

uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Use tempfile to create temporary files securely
    # The 'with' statement ensures that the temporary directory and its contents are removed automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video_path = os.path.join(tmpdir, uploaded_file.name)
        temp_audio_path = os.path.join(tmpdir, f"{os.path.splitext(uploaded_file.name)[0]}.wav")

        # Save the uploaded video file to a temporary location
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Video saved temporarily: {temp_video_path}")

        # Ensure FFmpeg is ready before proceeding
        with st.spinner("Checking FFmpeg setup..."):
            if not ensure_ffmpeg_access():
                st.error("Cannot proceed without a working FFmpeg setup. Please resolve the FFmpeg issue.")
                st.stop() # Stop execution if ffmpeg is not ready

        # Extract audio
        with st.spinner("Extracting audio from video..."):
            # Assign the return value to a variable, which could be None if failed
            extracted_audio_path = extract_audio_pydub(temp_video_path, temp_audio_path)
            
            # Crucial check: Stop if audio extraction failed (i.e., extracted_audio_path is None)
            if extracted_audio_path is None: 
                st.error("Audio extraction failed. Please check the video file format and ensure FFmpeg is functioning correctly.")
                st.stop() # Stop execution if audio extraction fails

        # Load model
        with st.spinner(f"Loading Faster-Whisper model: {model_size} ..."):
            model = load_model(model_size=model_size)

        # Transcribe
        with st.spinner("Transcribing (word timestamps enabled)... This may take a while depending on audio length and model size."):
            # CHANGED: Force transcription to English (lang="en")
            words = transcribe_words(_model=model, audio_path=extracted_audio_path, lang="en") 
            if not words:
                st.error("No words were detected. Please try a clearer audio or a different model size.")
                st.stop()

        # Build SRT by duration buckets
        with st.spinner("Building SRT by fixed duration and character limits..."):
            srt_text = bucket_words_by_duration(words, bucket_seconds=bucket_seconds, max_chars_per_sub=max_chars)

        st.success("Done!")
        st.subheader("Preview (SRT)")
        st.text_area("SRT Content", srt_text, height=320)

        # Download button
        base_name = os.path.splitext(uploaded_file.name)[0]
        dl_name = f"{base_name}_english_sub.srt" # Changed default download name to reflect English
        st.download_button(
            "Download SRT",
            data=srt_text.encode("utf-8"),
            file_name=dl_name,
            mime="text/plain"
        )
