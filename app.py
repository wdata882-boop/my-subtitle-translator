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
st.subheader("Translate with AI!") # Removed Myanmar language specific text

st.markdown("""
    This app uses the **Faster-Whisper** model to generate word-level timestamps, 
    and then converts them into a standard SRT subtitle file.
    It supports multiple languages.
    
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
# Changed default value to 'en' for English and removed Myanmar hint
lang_hint = st.sidebar.text_input("Language Hint (e.g., 'en' for English, 'auto' for auto-detect)", value="en")

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
    # Initialize temp_audio_path and temp_input_path at the beginning of the function
    # This helps ensure they are always defined for cleanup in the finally block
    temp_audio_path = None 
    temp_input_path = None 
    temp_dir = None # Initialize temp_dir as well

    uploaded = st.file_uploader("Upload Audio/Video", type=["mp3", "mp4", "wav", "m4a", "flac", "mov", "avi", "mkv"])

    try:
        if uploaded:
            st.success("File uploaded successfully!")
            
            # Create a temporary directory for all temporary files
            temp_dir = tempfile.mkdtemp()
            st.info(f"Temporary directory created at: {temp_dir}")
            
            file_extension = uploaded.name.split('.')[-1].lower()
            temp_input_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{file_extension}")
            
            with open(temp_input_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.info(f"Processing uploaded file: {uploaded.name}")

            # Extract audio if it's a video file
            if file_extension in ["mp4", "mov", "avi", "mkv"]:
                st.info("Video file detected. Extracting audio...")
                if ensure_ffmpeg():
                    with st.spinner("Extracting audio..."):
                        # Output audio file will also be in the temp_dir
                        temp_audio_path = extract_audio_ffmpeg(temp_input_path, os.path.join(temp_dir, f"{uuid.uuid4()}.wav"))
                        if not temp_audio_path:
                            st.error("Failed to extract audio from the video. Please check file format and FFmpeg installation.")
                            return # Stop execution to prevent further errors
                else:
                    st.error("FFmpeg is required to extract audio from video files. Please ensure it's installed and accessible.")
                    return # Stop execution to prevent further errors
            elif file_extension in ["mp3", "wav", "m4a", "flac"]:
                temp_audio_path = temp_input_path # If it's an audio file, treat it directly as audio path
            else:
                st.error("Unsupported file format. Please upload an audio (mp3, wav, m4a, flac) or video (mp4, mov, avi, mkv) file.")
                return # Stop execution to prevent further errors
            
            if temp_audio_path: # temp_audio_path is now guaranteed to be defined if we reach here
                # Load model
                with st.spinner(f"Loading Faster-Whisper model: {model_size} ..."):
                    model = load_model(model_size=model_size)

                # Transcribe
                with st.spinner("Transcribing (word timestamps enabled)... This may take a while."):
                    lang = lang_hint.strip() if lang_hint.strip() != "auto" else None # If 'auto', let Whisper detect
                    words = transcribe_words(temp_audio_path, model, lang=lang)
                    if not words:
                        st.error("No words were detected. Please try a clearer audio or a different model size.")
                        return # Stop execution to prevent further errors

                # Build SRT by duration buckets
                with st.spinner("Building SRT by fixed duration..."):\
                    srt_text = bucket_words_by_duration(words, bucket_seconds=bucket_seconds, max_chars_per_sub=max_chars)

                st.success("Done!")
                st.subheader("Preview (SRT)")
                st.text_area("SRT Content", srt_text, height=320)

                # Prepare for download
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
        # Final cleanup for all temporary files and directory
        # This block runs after processing, even if errors occurred (except for st.stop() paths, which we replaced with return)
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
                os.rmdir(temp_dir) # Use rmdir only if directory is empty
                st.info(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                # If directory is not empty (e.g., if model was downloaded there by mistake)
                st.warning(f"Could not remove temporary directory {temp_dir}. It might not be empty: {e}")


if __name__ == "__main__":
    main()