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
        # Capture stderr and stdout to see if there's any output
        result = subprocess.run(["ffmpeg", "-version"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                check=True)
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


def extract_audio_ffmpeg(input_path: str, output_path: str, sr: int = 16000) -> None:
    """
    Use ffmpeg to extract mono wav @16kHz for transcription stability.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",          # No video
        "-ac", "1",     # Mono audio
        "-ar", str(sr), # Set audio sample rate
        "-map_metadata", "-1", # Remove metadata
        "-acodec", "pcm_s16le", # PCM 16-bit signed-integer little-endian
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg audio extraction failed: {e.stderr.decode()}")
        raise # Re-raise the exception to be caught by the calling function


@st.cache_resource # This caches the model, so it's loaded only once per session
def load_model(model_size: str = "tiny") -> WhisperModel:
    """
    Loads the Faster-Whisper model.
    """
    st.info(f"Loading Faster-Whisper '{model_size}' model. This may take a moment...")
    try:
        # device="cpu" for Streamlit Cloud as GPUs are not typically available
        # compute_type="int8" for CPU optimization
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        st.success(f"Faster-Whisper '{model_size}' model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading Faster-Whisper model: {e}")
        st.stop()


def transcribe_words(audio_path: str, model: WhisperModel, lang: str = None) -> list:
    """
    Transcribes audio and returns word-level timestamps.
    """
    segments, info = model.transcribe(audio_path, word_timestamps=True, language=lang)
    all_words = []
    for segment in segments:
        for word in segment.words:
            all_words.append({
                "start": word.start,
                "end": word.end,
                "text": word.word.strip()
            })
    return all_words

def bucket_words_by_duration(words: list, bucket_seconds: float = 3.0, max_chars_per_sub: int = 40) -> str:
    """
    Groups words into SRT subtitle buckets based on a fixed duration and max characters.
    """
    srt_lines = []
    current_bucket_text = ""
    current_bucket_start_time = None
    subtitle_index = 1
    
    # Iterate through each word and try to fit it into the current bucket
    for i, word_data in enumerate(words):
        word_text = word_data["text"]
        word_start = word_data["start"]
        word_end = word_data["end"]
        
        # Initialize current bucket start time if it's the first word or after a new bucket starts
        if current_bucket_start_time is None:
            current_bucket_start_time = word_start
        
        # Calculate potential end time of the current bucket if this word is added
        potential_bucket_end_time = word_end
        
        # Check if adding this word exceeds max_chars_per_sub or bucket_seconds
        # Or if it's the last word
        if (len(current_bucket_text + " " + word_text) > max_chars_per_sub and current_bucket_text != "") or \
           (potential_bucket_end_time - current_bucket_start_time > bucket_seconds) or \
           (i == len(words) - 1):
            
            # If current bucket is not empty, finalize it before starting a new one
            if current_bucket_text.strip() != "":
                # For the last word, ensure the bucket ends at the word's end time
                end_time_for_bucket = potential_bucket_end_time if i == len(words) - 1 else current_bucket_start_time + bucket_seconds

                srt_lines.append(str(subtitle_index))
                srt_lines.append(f"{hhmmss_ms(current_bucket_start_time)} --> {hhmmss_ms(end_time_for_bucket)}")
                srt_lines.append(current_bucket_text.strip())
                srt_lines.append("") # Empty line for SRT format
                subtitle_index += 1
            
            # Start a new bucket with the current word
            current_bucket_text = word_text
            current_bucket_start_time = word_start # New bucket starts at the current word's start
        else:
            # Add word to current bucket
            if current_bucket_text:
                current_bucket_text += " " + word_text
            else:
                current_bucket_text = word_text
                
    # Handle the very last bucket if it wasn't added in the loop
    if current_bucket_text.strip() != "" and (not srt_lines or srt_lines[-1] != ""): # Corrected condition for last bucket
        srt_lines.append(str(subtitle_index))
        srt_lines.append(f"{hhmmss_ms(current_bucket_start_time)} --> {hhmmss_ms(words[-1]['end'])}")
        srt_lines.append(current_bucket_text.strip())
        srt_lines.append("")

    return "\n".join(srt_lines)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Universal Subtitle Translator", layout="centered")

st.title("🎬 Universal Subtitle Translator")
st.markdown("Easily convert Video/Audio files into **English Subtitles** using AI!")

st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 0.5rem;
    padding: 0.8rem;
    font-size: 1.1rem;
    background-color: #4CAF50;
    color: white;
    border: none;
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #45a049;
}
.stFileUploader>div>div {
    border-radius: 0.5rem;
    border: 2px dashed #4CAF50;
    padding: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize temp paths
temp_video_path = None
temp_audio_path = None # Ensure this is initialized

# Model size selection
model_size = st.selectbox("Choose Whisper Model Size (smaller is faster):", ["tiny", "base", "small"], index=0)
lang_hint = st.text_input("Source Language Hint (e.g., 'Burmese', leave blank for auto-detect):", value="")
bucket_seconds = st.slider("Max Subtitle Duration (seconds):", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
max_chars = st.slider("Max Characters Per Subtitle Line:", min_value=20, max_value=80, value=40, step=5)


uploaded = st.file_uploader("Upload an audio or video file", type=["mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a"])

if uploaded is not None:
    # 1. Save uploaded file to temp path
    suffix = os.path.splitext(uploaded.name)[1]
    # Use mkstemp for secure temporary file creation
    fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(uploaded.read()) # Use .read() for byte content directly
    
    st.info(f"Processing '{uploaded.name}'...")
    
    # Determine if it's an audio or video file
    mime_type = uploaded.type
    is_video = mime_type.startswith("video/")

    # Initialize wav_path. If it's an audio file, it's the temp_file_path itself.
    # If it's a video, temp_audio_path will be generated.
    wav_path = temp_file_path 

    # 2. Extract audio if it's a video
    if is_video:
        # Define temp_audio_path only if it's a video and we need to extract audio
        if not os.path.exists("temp"): # Ensure 'temp' directory exists for audio extraction
            os.makedirs("temp")
        temp_audio_path = os.path.join("temp", f"{uuid.uuid4()}.wav") # This defines temp_audio_path
        with st.spinner("Extracting audio from video..."):
            try:
                # Call ffmpeg to extract audio
                extract_audio_ffmpeg(temp_file_path, temp_audio_path)
                wav_path = temp_audio_path # Now wav_path points to the extracted audio
            except Exception as e:
                st.error(f"Error during audio extraction: {e}. Please try another file or ensure FFmpeg is correctly set up.")
                # Clean up temp files if extraction fails
                os.remove(temp_file_path)
                # Only try to remove temp_audio_path if it was actually defined (i.e., if it was a video)
                if temp_audio_path and os.path.exists(temp_audio_path): # Check if temp_audio_path is not None and exists
                    os.remove(temp_audio_path)
                st.stop() # Stop further execution if audio extraction fails
    
    # Ensure FFmpeg is available (this check is important after file upload)
    if not ensure_ffmpeg():
        st.error("FFmpeg not found. Please ensure it's installed correctly on the server.")
        # Clean up temp files before stopping
        os.remove(temp_file_path)
        if is_video and temp_audio_path and os.path.exists(temp_audio_path): # Ensure temp_audio_path is defined
            os.remove(temp_audio_path)
        st.stop()

    # Load model
    with st.spinner(f"Loading Faster-Whisper model: {model_size} ..."):
        model = load_model(model_size=model_size)

    # Transcribe
    with st.spinner("Transcribing (word timestamps enabled)... This may take a while for large files."):
        lang = lang_hint.strip() if lang_hint.strip() else None
        # Use the correct path for transcription
        words = transcribe_words(wav_path, model, lang=lang) 
        if not words:
            st.error("No words were detected. Please try a clearer audio or a different model size.")
            # Clean up temp files
            os.remove(temp_file_path)
            if is_video and temp_audio_path and os.path.exists(temp_audio_path): # Ensure temp_audio_path is defined
                os.remove(temp_audio_path)
            st.stop()

    # Build SRT by duration buckets
    with st.spinner("Building SRT by fixed duration..."):
        srt_text = bucket_words_by_duration(words, bucket_seconds=bucket_seconds, max_chars_per_sub=max_chars)

    st.success("Done!")
    st.subheader("Preview (SRT)")
    st.text_area("SRT Content", srt_text, height=320)

    # Correct filename for download
    base_name = os.path.splitext(uploaded.name)[0] # Just the name, no extension
    dl_name = f"{base_name}_translated_{int(bucket_seconds*1000)}ms.srt" # Consistent naming
    st.download_button(
        "Download SRT",
        data=srt_text.encode("utf-8"),
        file_name=dl_name,
        mime="text/plain"
    )

    # Clean up temporary files finally
    os.remove(temp_file_path)
    # This is the final cleanup. Ensure temp_audio_path is defined and exists before trying to remove.
    if is_video and temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

st.markdown("---")
st.markdown("Developed with ❤️ using Faster-Whisper and Streamlit.")