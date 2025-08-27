# -*- coding: utf-8 -*-
import os
import requests
import tempfile
import subprocess
import streamlit as st

# Use faster-whisper for word-level timestamps
from faster_whisper import WhisperModel

# Use pydub for audio extraction
from pydub import AudioSegment

# ----------------------------
# Constants and UI Configuration
# ----------------------------
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL_SIZE = "base"
MAX_CHARS_PER_SUB = 60

# ----------------------------
# Helper Functions
# ----------------------------
def hhmmss_ms(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm"""
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

@st.cache_resource
def load_model(model_size: str):
    """Loads the Faster-Whisper model. Uses cache for efficiency."""
    st.info(f"Loading Faster-Whisper model: {model_size}. This may take a while...")
    try:
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        st.error("This could be due to a temporary issue with downloading the model. Please try reloading the page.")
        return None

def translate_text_mymemory(text: str, source_lang: str, target_lang: str = "en") -> str:
    """Translates text using the free MyMemory API."""
    if not text.strip():
        return ""
    
    api_url = "https://api.mymemory.translated.net/get"
    params = {
        'q': text,
        'langpair': f'{source_lang}|{target_lang}'
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        response_json = response.json()
        
        translated_text = response_json['responseData']['translatedText']
        return translated_text
    except Exception as e:
        st.warning(f"Translation failed due to an API error: {e}. Returning original text.")
        return text

def transcribe_and_get_segments(_model: WhisperModel, audio_path: str):
    """
    Transcribes audio and returns segments with timestamps and detected language info.
    """
    st.info("Transcribing audio... This might take some time.")
    segments, info = _model.transcribe(audio_path, beam_size=5)
    
    st.success(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    
    return list(segments), info.language

def create_srt_from_segments(segments: list, source_language: str) -> str:
    """
    Creates SRT content from transcribed segments, with translation using MyMemory API.
    """
    srt_content = []
    
    with st.spinner("Translating text segments to English via MyMemory API..."):
        for i, segment in enumerate(segments):
            original_text = segment.text.strip()
            
            # Translate the segment's text to English
            translated_text = translate_text_mymemory(original_text, source_lang=source_language, target_lang="en")

            # Line breaking for long subtitles
            if len(translated_text) > MAX_CHARS_PER_SUB:
                break_point = translated_text.rfind(' ', 0, MAX_CHARS_PER_SUB)
                if break_point != -1:
                    line1 = translated_text[:break_point]
                    line2 = translated_text[break_point+1:]
                    final_text = f"{line1.strip()}\n{line2.strip()}"
                else:
                    final_text = f"{translated_text[:MAX_CHARS_PER_SUB]}\n{translated_text[MAX_CHARS_PER_SUB:]}"
            else:
                final_text = translated_text

            start_time_str = hhmmss_ms(segment.start)
            end_time_str = hhmmss_ms(segment.end)
            
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time_str} --> {end_time_str}")
            srt_content.append(final_text)
            srt_content.append("")
            
    return "\n".join(srt_content)

def ensure_ffmpeg_access():
    """Ensures ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error("FFmpeg not found. It's required for audio extraction. Ensure it's in your packages.txt for Streamlit Cloud.")
        return False

def extract_audio_pydub(input_path: str, output_path: str) -> str:
    """Uses pydub to extract mono wav @16kHz."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        st.success(f"Audio extracted successfully to {output_path}")
        return output_path
    except Exception as e:
        st.error(f"Audio extraction failed: {e}. Check if the uploaded file is a valid media format.")
        return None

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(layout="wide", page_title="AI Subtitle Generator")

st.title("AI Video to Subtitle Generator 🎬")
st.markdown("""
    Upload any video file. This app will:
    1.  **Extract the audio** using FFmpeg.
    2.  **Detect the language** and transcribe it to text using a Whisper model.
    3.  **Translate the text to English** using the free MyMemory API.
    4.  **Generate an SRT subtitle file** with accurate timings.
""")

st.sidebar.header("Configuration")
model_size = st.sidebar.selectbox("Whisper Model Size", MODEL_SIZES, index=MODEL_SIZES.index(DEFAULT_MODEL_SIZE),
                                  help="Larger models are more accurate but slower. 'base' is a good balance.")

st.sidebar.info("This app uses the MyMemory API for free translation. No API key is required.")

uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video_path = os.path.join(tmpdir, uploaded_file.name)
        temp_audio_path = os.path.join(tmpdir, f"{os.path.splitext(uploaded_file.name)[0]}.wav")

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if not ensure_ffmpeg_access():
            st.stop()

        extracted_audio_path = extract_audio_pydub(temp_video_path, temp_audio_path)
        if not extracted_audio_path:
            st.stop()

        model = load_model(model_size=model_size)
        if not model:
            st.stop()
            
        segments, detected_language = transcribe_and_get_segments(_model=model, audio_path=extracted_audio_path)
        if not segments:
            st.error("No text could be transcribed from the audio. The audio might be silent or too noisy.")
            st.stop()

        st.subheader(f"Original Transcription (Detected Language: {detected_language.upper()})")
        original_text_preview = " ".join([seg.text.strip() for seg in segments])
        st.text_area("Original Text", original_text_preview, height=150)

        srt_text = create_srt_from_segments(segments, source_language=detected_language)

        st.success("SRT Subtitle Generation Complete!")
        st.subheader("Translated English Subtitles (SRT Preview)")
        st.text_area("SRT Content", srt_text, height=300)

        base_name = os.path.splitext(uploaded_file.name)[0]
        dl_name = f"{base_name}_english_subtitles.srt"
        st.download_button(
            label="Download .SRT File",
            data=srt_text.encode("utf-8"),
            file_name=dl_name,
            mime="text/plain"
        )
