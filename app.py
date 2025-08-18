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
DEFAULT_MODEL_SIZE = "small"
MAX_CHARS_PER_SUB = 60 # Maximum characters per subtitle line
DEFAULT_BUCKET_SECONDS = 5 # Default duration for subtitle buckets (maximum duration)
DEFAULT_MIN_SUBTITLE_DURATION = 0.5 # Default minimum duration for a subtitle entry

# FFmpeg will be installed via packages.txt and should be in the system's PATH
# No need to specify a local path like "./ffmpeg"

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
    # Changed compute_type to "int8" for better CPU compatibility on Streamlit Cloud
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
        beam_size=7 
    )
    words = []
    for segment in segments:
        for word in segment.words:
            words.append(word)
    return words

def bucket_words_by_duration(words: list, bucket_seconds: int = 5, max_chars_per_sub: int = 60, min_subtitle_duration: float = 0.5) -> str:
    """
    Creates SRT content by segmenting based on natural pauses (punctuation)
    and then falling back to duration/character limits if needed.
    Ensures each subtitle meets a minimum display duration.
    """
    srt_content = []
    subtitle_idx = 1
    current_words_in_segment = []

    # Helper to check if a word ends with punctuation
    def ends_with_punctuation(word_text):
        # Added common English and Chinese punctuation
        return word_text.endswith(('.', '?', '!', ',', '。', '？', '！', '，'))

    for i, word in enumerate(words):
        # Using word.word as determined by previous error logs for compatibility
        word_text = word.word 

        current_words_in_segment.append(word)
        
        # Calculate current segment's properties
        # CHANGED: Added " " to join words with spaces
        segment_text = " ".join([w.word for w in current_words_in_segment]).strip()
        
        # Ensure there are words in the segment before calculating duration
        segment_duration = 0.0
        if current_words_in_segment:
             segment_duration = current_words_in_segment[-1].end - current_words_in_segment[0].start
        
        segment_char_length = len(segment_text)

        # Break conditions:
        punctuation_break = False
        if ends_with_punctuation(word_text):
            # If it's a period/question mark/exclamation mark, prefer to break
            if word_text.endswith(('.', '?', '!', '。', '？', '！')):
                punctuation_break = True
            # If it's a comma, break if segment is reasonably long
            elif word_text.endswith((',', '，')):
                if segment_duration >= (min_subtitle_duration + 0.5) or segment_char_length >= (max_chars_per_sub / 2):
                    punctuation_break = True
            
            # Additional check: If the gap to the next word is large, it's a good break point
            if i < len(words) - 1:
                next_word_start = words[i+1].start
                if (next_word_start - word.end) > 0.4: # A significant pause (e.g., 0.4 seconds)
                    punctuation_break = True

        duration_exceeded = segment_duration >= bucket_seconds
        chars_exceeded = segment_char_length >= max_chars_per_sub
        is_last_word = (i == len(words) - 1)

        # Finalize segment if any break condition is met, and it's not too short (unless it's the very last word)
        should_finalize = False
        if is_last_word:
            should_finalize = True
        elif punctuation_break and segment_duration >= min_subtitle_duration:
            should_finalize = True
        elif duration_exceeded and segment_duration >= min_subtitle_duration:
            should_finalize = True
        elif chars_exceeded and segment_duration >= min_subtitle_duration: # Prioritize chars_exceeded only if min_duration met
            should_finalize = True
        
        # Special case: If we're forcing a break due to max_chars_per_sub,
        # ensure it's not an empty string or just spaces
        if should_finalize and segment_text.strip() == "" and not is_last_word:
            should_finalize = False # Don't finalize an empty segment

        if should_finalize:
            if current_words_in_segment:
                start_time = current_words_in_segment[0].start
                end_time = current_words_in_segment[-1].end + 0.02
                
                # Apply minimum duration for display
                if (end_time - start_time) < min_subtitle_duration:
                    end_time = start_time + min_subtitle_duration

                # CHANGED: Added " " to join words with spaces
                subtitle_text = " ".join([w.word.strip() for w in current_words_in_segment]).strip()
                
                # Line breaking for subtitles that are still too long after segmentation
                if len(subtitle_text) > max_chars_per_sub:
                    break_point = -1
                    # Try to break at a punctuation mark within the max_chars limit from the end
                    for k in range(min(len(subtitle_text) -1, max_chars_per_sub -1), -1, -1):
                        if subtitle_text[k] in (',', '.', '?', '!', '。', '？', '！', '，'):
                            break_point = k + 1
                            break
                    if break_point == -1: # No punctuation, try last space
                        break_point = subtitle_text.rfind(' ', 0, max_chars_per_sub)
                    if break_point == -1: # No space either, force break at max_chars
                        break_point = max_chars_per_sub
                    
                    line1 = subtitle_text[:break_point].strip()
                    line2 = subtitle_text[break_point:].strip()
                    subtitle_text_final = f"{line1}\n{line2}"
                else:
                    subtitle_text_final = subtitle_text
                
                if subtitle_text_final: # Only add if text is not empty after stripping
                    start_time_str = hhmmss_ms(start_time)
                    end_time_str = hhmmss_ms(end_time)
                    
                    srt_content.append(f"{subtitle_idx}")
                    srt_content.append(f"{start_time_str} --> {end_time_str}")
                    srt_content.append(subtitle_text_final)
                    srt_content.append("")
                    subtitle_idx += 1
                
                current_words_in_segment = [] # Reset for next segment

    # Final check for any remaining words (should be caught by is_last_word, but as a safeguard)
    if current_words_in_segment:
        start_time = current_words_in_segment[0].start
        end_time = current_words_in_segment[-1].end + 0.02
        if (end_time - start_time) < min_subtitle_duration:
            end_time = start_time + min_subtitle_duration

        # CHANGED: Added " " to join words with spaces
        final_segment_text = " ".join([w.word.strip() for w in current_words_in_segment]).strip()
        if len(final_segment_text) > max_chars_per_sub:
            break_point = -1
            for k in range(min(len(final_segment_text) -1, max_chars_per_sub -1), -1, -1):
                if final_segment_text[k] in (',', '.', '?', '!', '。', '？', '！', '，'):
                    break_point = k + 1
                    break
            if break_point == -1:
                break_point = final_segment_text.rfind(' ', 0, max_chars_per_sub)
            if break_point == -1:
                break_point = max_chars_per_sub
            
            line1 = final_segment_text[:break_point].strip()
            line2 = final_segment_text[break_point:].strip()
            final_subtitle_text = f"{line1}\n{line2}"
        else:
            final_subtitle_text = final_segment_text
        
        if final_subtitle_text:
            start_time_str = hhmmss_ms(start_time)
            end_time_str = hhmmss_ms(end_time)
            srt_content.append(f"{subtitle_idx}")
            srt_content.append(f"{start_time_str} --> {end_time_str}")
            srt_content.append(final_subtitle_text)
            srt_content.append("")

    return "\n".join(srt_content)

def ensure_ffmpeg_access():
    """
    Ensures the ffmpeg executable is available in the system PATH.
    It will be installed via packages.txt during deployment.
    """
    try:
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
        AudioSegment.converter = "ffmpeg"
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(sr)  # 16kHz
        audio.export(output_path, format="wav")
        st.success(f"Audio extracted successfully to {output_path}")
        return output_path
    except Exception as e:
        st.error(f"Error extracting audio with pydub (FFmpeg issue?): {e}. Please check your video file format and ensure 'ffmpeg' is installed correctly via packages.txt.")
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
    3. Generate an SRT subtitle file, intelligently segmenting lines based on punctuation, pauses, and configurable duration/character limits.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_size = st.sidebar.selectbox("Choose Whisper Model Size", MODEL_SIZES, index=MODEL_SIZES.index(DEFAULT_MODEL_SIZE))

# Maximum subtitle duration slider
bucket_seconds = st.sidebar.slider(
    "Maximum Subtitle Duration (seconds)", 
    min_value=1, 
    max_value=10, 
    value=DEFAULT_BUCKET_SECONDS,
    help="Sets the maximum duration for a single subtitle entry. Shorter durations mean more frequent subtitle changes."
)

# Minimum subtitle duration slider
min_duration_seconds = st.sidebar.slider(
    "Minimum Subtitle Duration (seconds)", 
    min_value=0.1, 
    max_value=2.0, 
    value=DEFAULT_MIN_SUBTITLE_DURATION, 
    step=0.1, 
    help="Ensures each subtitle is displayed for at least this minimum duration. Prevents very short, flickering subtitles."
)

# Max Characters Per Subtitle Line slider
max_chars = st.sidebar.slider(
    "Max Characters Per Subtitle Line", 
    min_value=20, 
    max_value=100, 
    value=MAX_CHARS_PER_SUB,
    help="Sets the maximum number of characters allowed on a single line of subtitle text. Text exceeding this limit will be wrapped to the next line."
)

uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video_path = os.path.join(tmpdir, uploaded_file.name)
        temp_audio_path = os.path.join(tmpdir, f"{os.path.splitext(uploaded_file.name)[0]}.wav")

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Video saved temporarily: {temp_video_path}")

        with st.spinner("Checking FFmpeg setup..."):
            if not ensure_ffmpeg_access():
                st.error("Cannot proceed without a working FFmpeg setup. Please resolve the FFmpeg issue.")
                st.stop()

        with st.spinner("Extracting audio from video..."):
            extracted_audio_path = extract_audio_pydub(temp_video_path, temp_audio_path)
            if extracted_audio_path is None: 
                st.error("Audio extraction failed. Please check the video file format and ensure FFmpeg is functioning correctly.")
                st.stop()

        with st.spinner(f"Loading Faster-Whisper model: {model_size} ..."):
            model = load_model(model_size=model_size)

        with st.spinner("Transcribing (word timestamps enabled)... This may take a while depending on audio length and model size."):
            words = transcribe_words(_model=model, audio_path=extracted_audio_path, lang="en") 
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
        dl_name = f"{base_name}_english_sub.srt" 
        st.download_button(
            "Download SRT",
            data=srt_text.encode("utf-8"),
            file_name=dl_name,
            mime="text/plain"
        )
