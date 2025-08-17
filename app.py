# -*- coding: utf-8 -*-
import os
import uuid
import math
import tempfile
import subprocess
from datetime import timedelta
import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg_python as ffmpeg

# ----------------------------
# Constants and UI Configuration
# ----------------------------

# Set Streamlit page configuration
st.set_page_config(
    page_title="ရုပ်ရှင် သို့မဟုတ် ဗီဒီယို ဖိုင်များမှ စာတန်းထိုးများ ဖန်တီးပါ။",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 ဇာတ်ကား သို့မဟုတ် ဗီဒီယိုမှ စာတန်းထိုးများ ဖန်တီးရန်")
st.markdown("အောက်ပါ အချက်အလက်များအတိုင်း ဗီဒီယို သို့မဟုတ် အသံဖိုင်မှ စာတန်းထိုးဖိုင် (SRT) ကို အလိုအလျောက် ထုတ်ပေးနိုင်ပါတယ်။")

# ----------------------------
# Helpers
# ----------------------------

def hhmmss_ms(seconds: float) -> str:
    """
    SRT time format HH:MM:SS,mmm
    Converts a float number of seconds into an SRT-formatted timestamp string.
    """
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
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_ffmpeg(input_path: str, output_path: str, sr: int = 16000) -> None:
    """
    Uses ffmpeg to extract mono WAV audio @16kHz for transcription stability.
    """
    try:
        # Construct the ffmpeg command using the ffmpeg-python library
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar=sr)
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        st.error(f"Error extracting audio: {e.stderr.decode('utf-8')}")
        raise


def load_model(model_size: str = "base"):
    """
    Loads the Faster-Whisper model, using Streamlit's persistent cache.
    This resolves the 'read-only file system' error.
    """
    try:
        # Use Streamlit's recommended persistent cache location
        model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "faster-whisper-models")
        os.makedirs(model_cache_dir, exist_ok=True)
        st.info(f"Model cache directory: {model_cache_dir}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=model_cache_dir)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def transcribe_words(audio_path: str, model: WhisperModel, lang: str = None):
    """
    Transcribes audio to get word-level timestamps using faster-whisper.
    """
    try:
        segments, info = model.transcribe(audio_path, language=lang, word_timestamps=True, beam_size=5)
        st.info(f"Detected language: {info.language} with probability {info.language_probability:.4f}")
        all_words = []
        for segment in segments:
            if segment.words:
                all_words.extend(segment.words)
        return all_words
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None


def bucket_words_by_duration(words, bucket_seconds: float, max_chars_per_sub: int) -> str:
    """
    Builds an SRT string by grouping words into time-based buckets.
    Ensures that each subtitle line doesn't exceed a maximum character count.
    """
    srt_text = ""
    current_bucket = []
    current_bucket_end = 0.0
    subtitle_index = 1

    for word in words:
        if word.start >= current_bucket_end:
            # New bucket
            if current_bucket:
                # Add previous bucket to SRT
                start_time = current_bucket[0].start
                end_time = current_bucket[-1].end
                text = " ".join([w.word.strip() for w in current_bucket])
                if len(text) > max_chars_per_sub:
                    # If the subtitle is too long, split it
                    words_in_text = text.split()
                    temp_line = ""
                    for i, w in enumerate(words_in_text):
                        if len(temp_line) + len(w) + 1 > max_chars_per_sub and temp_line:
                            srt_text += f"{subtitle_index}\n"
                            srt_text += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(current_bucket[i-1].end)}\n"
                            srt_text += temp_line.strip() + "\n\n"
                            subtitle_index += 1
                            temp_line = w
                            start_time = current_bucket[i].start
                        else:
                            temp_line += " " + w if temp_line else w
                    if temp_line:
                        srt_text += f"{subtitle_index}\n"
                        srt_text += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(end_time)}\n"
                        srt_text += temp_line.strip() + "\n\n"
                        subtitle_index += 1
                else:
                    srt_text += f"{subtitle_index}\n"
                    srt_text += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(end_time)}\n"
                    srt_text += text.strip() + "\n\n"
                    subtitle_index += 1
                current_bucket = []

            # Start a new bucket
            bucket_start_time = math.floor(word.start / bucket_seconds) * bucket_seconds
            current_bucket_end = bucket_start_time + bucket_seconds
        
        current_bucket.append(word)

    # Add the last remaining bucket
    if current_bucket:
        start_time = current_bucket[0].start
        end_time = current_bucket[-1].end
        text = " ".join([w.word.strip() for w in current_bucket])
        if len(text) > max_chars_per_sub:
            words_in_text = text.split()
            temp_line = ""
            for i, w in enumerate(words_in_text):
                if len(temp_line) + len(w) + 1 > max_chars_per_sub and temp_line:
                    srt_text += f"{subtitle_index}\n"
                    srt_text += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(current_bucket[i-1].end)}\n"
                    srt_text += temp_line.strip() + "\n\n"
                    subtitle_index += 1
                    temp_line = w
                    start_time = current_bucket[i].start
                else:
                    temp_line += " " + w if temp_line else w
            if temp_line:
                srt_text += f"{subtitle_index}\n"
                srt_text += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(end_time)}\n"
                srt_text += temp_line.strip() + "\n\n"
        else:
            srt_text += f"{subtitle_index}\n"
            srt_text += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(end_time)}\n"
            srt_text += text.strip() + "\n\n"

    return srt_text


# ----------------------------
# Streamlit App UI and Logic
# ----------------------------

st.sidebar.header("ချိန်ညှိမှုများ")
# File uploader
uploaded_file = st.sidebar.file_uploader(
    "ဗီဒီယို သို့မဟုတ် အသံဖိုင် တင်ပါ။ (MP4, MKV, MP3, WAV)",
    type=["mp4", "mkv", "mp3", "wav"],
)

# Model size selection
model_size = st.sidebar.selectbox(
    "အသံ မှတ်သားဖို့အတွက် မော်ဒယ်အရွယ်အစား",
    ("base", "small", "medium"),
    index=0
)
st.sidebar.markdown("""
**မော်ဒယ်ရွေးချယ်မှုအကြောင်း:**
* `base`: အသေးဆုံးနှင့် အမြန်ဆုံး။
* `small`: အတော်အသင့် အရွယ်အစားနှင့် ပိုမိုတိကျသည်။
* `medium`: အကြီးဆုံး၊ အနှေးဆုံးနှင့် အတိကျဆုံး။
""")

# Language hint
lang_hint = st.sidebar.text_input(
    "ဘာသာစကား အရိပ်အမြွက် (ဥပမာ: en, my, ja)"
)
st.sidebar.markdown("ဘာသာစကားကို မသိရင် လွတ်ထားပါ။")


# Transcription parameters
bucket_seconds = st.sidebar.slider(
    "တစ်ကြောင်းအတွက် စက္ကန့်အများဆုံး",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="Each subtitle entry will cover a maximum of this duration."
)

max_chars = st.sidebar.slider(
    "တစ်ကြောင်းအတွက် စာလုံးအများဆုံး",
    min_value=20,
    max_value=120,
    value=40,
    step=5,
    help="The maximum number of characters allowed in a single subtitle line. Longer subtitles will be split into multiple lines."
)

# Main processing logic
if uploaded_file is not None:
    st.info("ဖိုင်ကို လက်ခံရရှိပါပြီ။ စတင်လုပ်ဆောင်ပါမည်။")

    # Save uploaded file to a temporary location
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Check if ffmpeg is available
        if not ensure_ffmpeg():
            st.error("FFmpeg ကို မတွေ့ရှိပါ။ ကျေးဇူးပြု၍ FFmpeg ကို ထည့်သွင်းပါ။")
            st.stop()
        
        # Extract audio from video
        temp_audio_path = os.path.join(temp_dir, "audio.wav")
        with st.spinner("အသံဖိုင်ကို ထုတ်ယူနေသည်..."):
            try:
                extract_audio_ffmpeg(temp_video_path, temp_audio_path)
            except Exception as e:
                st.error(f"အသံထုတ်ယူမှု မအောင်မြင်ပါ- {e}")
                st.stop()
            
        # Load model
        with st.spinner(f"မော်ဒယ် {model_size} ကို တင်နေသည်..."):
            model = load_model(model_size=model_size)

        # Transcribe
        with st.spinner("စာသားပြောင်းလဲမှု (စကားလုံးအချိန်မှတ်များဖြင့်)... အချိန်အနည်းငယ် ကြာနိုင်ပါတယ်။"):
            lang = lang_hint.strip() if lang_hint.strip() else None
            words = transcribe_words(temp_audio_path, model, lang=lang)
            if not words:
                st.error("စကားလုံးများ မတွေ့ရှိပါ။ ပိုမိုရှင်းလင်းသော အသံဖိုင် သို့မဟုတ် မတူညီသော မော်ဒယ်အရွယ်အစားကို စမ်းကြည့်ပါ။")
                st.stop()

        # Build SRT by duration buckets
        with st.spinner("သတ်မှတ်ထားသော စက္ကန့်အတိုင်း SRT ဖန်တီးနေသည်..."):
            srt_text = bucket_words_by_duration(words, bucket_seconds=bucket_seconds, max_chars_per_sub=max_chars)
        
        st.success("ပြီးပါပြီ။")
        st.subheader("အမြည်း (SRT)")
        st.text_area("SRT Content", srt_text, height=320)

        base_name = os.path.splitext(uploaded_file.name)[0]
        dl_name = f"{base_name}_duration_{int(bucket_seconds*1000)}ms.srt"
        st.download_button(
            "SRT ဖိုင်ကို ဒေါင်းလုဒ်လုပ်ပါ။",
            data=srt_text.encode("utf-8"),
            file_name=dl_name,
            mime="text/plain",
        )

