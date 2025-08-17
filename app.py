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
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def extract_audio_ffmpeg(input_path: str, output_path: str, sr: int = 16000) -> None:
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