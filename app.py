# -*- coding: utf-8 -*-
import os
import uuid
import math
import tempfile
import subprocess
from datetime import timedelta

import streamlit as st

# Use faster-whisper for word-level timestamps
<<<<<<< HEAD
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
=======
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
>>>>>>> 9c7fcaf46645dd2ba3a99b19253d5e0af28ef9d8
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def extract_audio_ffmpeg(input_path: str, output_path: str, sr: int = 16000) -> None:
<<<<<<< HEAD
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
    if current_bucket_text.strip() != "" and srt_lines[-1] != "": # Check if last bucket was already added or if there's text remaining
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

# Initialize temp paths globally or before the main logic
# These lines should be already present and correct
temp_video_path = None
temp_audio_path = None

# Model size selection and other UI elements
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
                if os.path.exists(temp_audio_path): # This check is now safe as temp_audio_path is defined above
                    os.remove(temp_audio_path)
                st.stop() # Stop further execution if audio extraction fails
    
    # Ensure FFmpeg is available (this check is important after file upload)
    if not ensure_ffmpeg():
        st.error("FFmpeg not found. Please ensure it's installed correctly on the server.")
        # Clean up temp files before stopping
        os.remove(temp_file_path)
        if is_video and os.path.exists(temp_audio_path): # temp_audio_path needs to be defined for this line to be safe
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
            if is_video and os.path.exists(temp_audio_path):
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
    if is_video and os.path.exists(temp_audio_path): # This is where temp_audio_path might be undefined if it was an audio file originally
        os.remove(temp_audio_path)

st.markdown("---")
st.markdown("Developed with ❤️ using Faster-Whisper and Streamlit.")
=======
    """
    Use ffmpeg to extract mono wav @16kHz for transcription stability.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",          # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


@st.cache_resource(show_spinner=False)
def load_model(model_size: str = "small", device: str = "auto", compute_type: str = "int8_float16"):
    """
    Cache the Whisper model in memory so it doesn't reload every run.
    - model_size: tiny, base, small, medium, large-v3 (depending on GPU/CPU)
    - compute_type: "int8_float16" is a good default for CPU/GPU mixed
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_words(audio_wav_path: str, model: WhisperModel, lang: str = None, beam_size: int = 5):
    """
    Run Faster-Whisper and return a flat list of words with timestamps.
    Each word: { "word": str, "start": float, "end": float }
    """
    segments, info = model.transcribe(
        audio_wav_path,
        task="transcribe" if lang else "detect-language",
        language=lang,              # if None, model attempts detection
        beam_size=beam_size,
        vad_filter=True,
        word_timestamps=True,       # crucial for duration-based bucketing
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                # Some models may return None for start/end occasionally; guard it
                if w.start is None or w.end is None:
                    # Skip words without timestamps
                    continue
                words.append({
                    "word": w.word.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                })
    return words


def bucket_words_by_duration(words, bucket_seconds=3.0, max_chars_per_sub=80):
    """
    Build SRT items by fixed duration buckets rather than using model segments.
    Strategy:
    - Iterate words chronologically
    - Start a new subtitle when:
        a) current bucket time >= bucket_seconds
        b) text length would exceed max_chars_per_sub (soft wrap)
    - Merge small residuals elegantly
    Returns: list of dicts: [{"index": 1, "start": s, "end": e, "text": t}, ...]
    """
    srt_items = []
    if not words:
        return srt_items

    bucket_start = words,[object Object],["start"]
    current_start = bucket_start
    current_end = words,[object Object],["end"]
    current_text = []

    def flush_item(start_t, end_t, text_tokens):
        text = " ".join([t for t in text_tokens if t])
        text = " ".join(text.split())  # normalize spaces
        if not text:
            return None
        return {
            "start": max(0.0, start_t),
            "end": max(start_t, end_t),
            "text": text
        }

    for w in words:
        word_str = w["word"]
        w_start, w_end = w["start"], w["end"]

        # Pre-calc length if we append this word
        candidate_text = (" ".join(current_text + [word_str])).strip()
        bucket_elapsed = w_end - current_start

        # Check conditions to wrap
        need_wrap = False
        if bucket_elapsed >= bucket_seconds:
            need_wrap = True
        elif len(candidate_text) > max_chars_per_sub:
            need_wrap = True

        if need_wrap and current_text:
            item = flush_item(current_start, current_end, current_text)
            if item:
                srt_items.append(item)
            # Start new bucket
            current_text = [word_str]
            current_start = w_start
            current_end = w_end
        else:
            current_text.append(word_str)
            current_end = w_end

    # Flush tail
    if current_text:
        item = flush_item(current_start, current_end, current_text)
        if item:
            srt_items.append(item)

    # Post-processing: Avoid overly short final snippet (e.g., < 0.6s) by merging back if possible
    if len(srt_items) >= 2 and (srt_items[-1]["end"] - srt_items[-1]["start"]) < 0.6:
        tail = srt_items.pop()
        srt_items[-1]["end"] = tail["end"]
        srt_items[-1]["text"] = (srt_items[-1]["text"] + " " + tail["text"]).strip()

    # Assign indices and format
    srt_lines = []
    for idx, item in enumerate(srt_items, start=1):
        s = hhmmss_ms(item["start"])
        e = hhmmss_ms(item["end"])
        t = item["text"]
        srt_lines.append(str(idx))
        srt_lines.append(f"{s} --> {e}")
        srt_lines.append(t)
        srt_lines.append("")

    return "\n".join(srt_lines)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Duration-based Subtitle Maker", layout="centered")
st.title("🎬 Duration-based Subtitle Maker")
st.markdown("Video/Audio → English subtitles (SRT) with fixed duration per line (not model segments).")

with st.expander("Settings", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox(
            "Model size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=2,
            help="Bigger models are more accurate but slower."
        )
        bucket_seconds = st.slider(
            "Duration per subtitle (seconds)",
            min_value=1.0, max_value=8.0, value=3.0, step=0.5,
            help="Each subtitle line will approximately fit into this duration."
        )
    with col2:
        max_chars = st.slider(
            "Max characters per subtitle",
            min_value=20, max_value=120, value=80, step=5,
            help="Soft limit to keep each line readable."
        )
        lang_hint = st.text_input(
            "Language hint (optional, e.g., 'my', 'en', 'ja')",
            value="",
            help="Leave blank to auto-detect."
        )

uploaded = st.file_uploader("Upload a video/audio file", type=["mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a"])

if uploaded is not None:
    # Save to temp
    tmp_dir = tempfile.mkdtemp(prefix="dur-sub-")
    src_path = os.path.join(tmp_dir, uploaded.name)
    with open(src_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Check ffmpeg
    if not ensure_ffmpeg():
        st.error("FFmpeg not found on system PATH. Please install FFmpeg and retry.")
        st.stop()

    # Extract audio wav@16k
    wav_path = os.path.join(tmp_dir, "audio_16k.wav")
    with st.spinner("Extracting audio with FFmpeg..."):
        try:
            extract_audio_ffmpeg(src_path, wav_path, sr=16000)
        except subprocess.CalledProcessError:
            st.error("Audio extraction failed. Check your file format and FFmpeg installation.")
            st.stop()

    # Load model
    with st.spinner(f"Loading Faster-Whisper model: {model_size} ..."):
        model = load_model(model_size=model_size)

    # Transcribe
    with st.spinner("Transcribing (word timestamps enabled)... This may take a while."):
        lang = lang_hint.strip() if lang_hint.strip() else None
        words = transcribe_words(wav_path, model, lang=lang)
        if not words:
            st.error("No words were detected. Please try a clearer audio or a different model size.")
            st.stop()

    # Build SRT by duration buckets
    with st.spinner("Building SRT by fixed duration..."):
        srt_text = bucket_words_by_duration(words, bucket_seconds=bucket_seconds, max_chars_per_sub=max_chars)

    st.success("Done!")
    st.subheader("Preview (SRT)")
    st.text_area("SRT Content", srt_text, height=320)

    base_name = os.path.splitext(uploaded.name),[object Object],
    dl_name = f"{base_name}_duration_{int(bucket_seconds*1000)}ms.srt"
    st.download_button(
        "Download SRT",
        data=srt_text.encode("utf-8"),
        file_name=dl_name,
        mime="text/plain"
    )

    # Cleanup notice (temp dir auto-deleted by OS later; we could add manual cleanup if needed)
    st.info("Tip: You can adjust duration/character limit and re-run with another file.")


st.markdown("---")
st.markdown("Developed with ❤️ using Faster-Whisper + FFmpeg + Streamlit.")
>>>>>>> 9c7fcaf46645dd2ba3a99b19253d5e0af28ef9d8
