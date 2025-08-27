# -*- coding: utf-8 -*-
import os
import math
import tempfile
import subprocess
import streamlit as st
import openai # pip install openai


# Use faster-whisper for word-level timestamps
# pip install faster-whisper
from faster_whisper import WhisperModel

# Use pydub for audio extraction
# pip install pydub
from pydub import AudioSegment

# ----------------------------
# Constants and UI Configuration
# ----------------------------
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL_SIZE = "base" # Consider "small" or "medium" for better quality if resources allow
MAX_CHARS_PER_SUB = 60 # Maximum characters per subtitle line
DEFAULT_BUCKET_SECONDS = 5 # Default duration for subtitle buckets (maximum duration)
DEFAULT_MIN_SUBTITLE_DURATION = 0.5 # Default minimum duration for a subtitle entry

# Ensure FFmpeg is accessible (it's installed via packages.txt on Streamlit Cloud)
# AudioSegment.converter = "ffmpeg" # Pydub finds it automatically if in PATH

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
# compute_type="int8" is generally best for CPU to avoid ValueError on Streamlit Cloud
return WhisperModel(model_size, device="cpu", compute_type="int8")

@st.cache_resource(hash_funcs={WhisperModel: lambda _: None}) # Exclude WhisperModel from hashing as it's unhashable
def transcribe_words(_model: WhisperModel, audio_path: str, lang: str = None):
"""
   Transcribes audio using Faster-Whisper with word-level timestamps.
   lang=None will auto-detect the language and transcribe in that language.
   """
st.info(f"Transcribing audio with word-level timestamps. Language set to: {lang if lang else 'Auto-detect'}")
segments, info = _model.transcribe(
audio_path,
word_timestamps=True, 
language=lang, # Set to None for auto-detection
beam_size=7 
)
words = []
for segment in segments:
for word in segment.words:
words.append(word)

detected_language = info.language
st.info(f"Detected language: {detected_language.upper()}")
return words, detected_language

def bucket_words_by_duration(words: list, bucket_seconds: int = 5, max_chars_per_sub: int = 60, min_subtitle_duration: float = 0.5) -> str:
"""
   Creates SRT content by segmenting based on natural pauses (punctuation)
   and then falling back to duration/character limits if needed.
   Ensures each subtitle meets a minimum display duration.
   """
srt_content_blocks = [] # To store list of dicts for each subtitle block
current_words_in_segment = []

def ends_with_punctuation(word_text):
# Handles common English and Chinese punctuation
return word_text.endswith(('.', '?', '!', ',', '。', '？', '！', '，'))

for i, word in enumerate(words):
word_text = word.word 
current_words_in_segment.append(word)

segment_text = " ".join([w.word for w in current_words_in_segment]).strip()

segment_duration = 0.0
if current_words_in_segment:
segment_duration = current_words_in_segment[-1].end - current_words_in_segment[0].start

segment_char_length = len(segment_text)

punctuation_break = False
if ends_with_punctuation(word_text):
if word_text.endswith(('.', '?', '!', '。', '？', '！')):
punctuation_break = True
elif word_text.endswith((',', '，')):
if segment_duration >= (min_subtitle_duration + 0.5) or segment_char_length >= (max_chars_per_sub / 2):
punctuation_break = True

if i < len(words) - 1:
next_word_start = words[i+1].start
if (next_word_start - word.end) > 0.4: # Significant pause
punctuation_break = True

duration_exceeded = segment_duration >= bucket_seconds
chars_exceeded = segment_char_length >= max_chars_per_sub
is_last_word = (i == len(words) - 1)

should_finalize = False
if is_last_word:
should_finalize = True
elif punctuation_break and segment_duration >= min_subtitle_duration:
should_finalize = True
elif duration_exceeded and segment_duration >= min_subtitle_duration:
should_finalize = True
elif chars_exceeded and segment_duration >= min_subtitle_duration:
should_finalize = True

if should_finalize and segment_text.strip() == "" and not is_last_word:
should_finalize = False

if should_finalize:
if current_words_in_segment:
start_time = current_words_in_segment[0].start
end_time = current_words_in_segment[-1].end + 0.02 # Add 20ms offset for smoother transition

if (end_time - start_time) < min_subtitle_duration:
end_time = start_time + min_subtitle_duration

subtitle_text = " ".join([w.word.strip() for w in current_words_in_segment]).strip()

# Line breaking for readability
if len(subtitle_text) > max_chars_per_sub:
break_point = -1
for k in range(min(len(subtitle_text) -1, max_chars_per_sub -1), -1, -1):
if subtitle_text[k] in (',', '.', '?', '!', '。', '？', '！', '，'):
break_point = k + 1
break
if break_point == -1:
break_point = subtitle_text.rfind(' ', 0, max_chars_per_sub)
if break_point == -1:
break_point = max_chars_per_sub

line1 = subtitle_text[:break_point].strip()
line2 = subtitle_text[break_point:].strip()
subtitle_text_final = f"{line1}\n{line2}"
else:
subtitle_text_final = subtitle_text

if subtitle_text_final:
srt_content_blocks.append({
"id": subtitle_idx,
"start": start_time,
"end": end_time,
"text": subtitle_text_final
})
subtitle_idx += 1

current_words_in_segment = [] # Reset for next segment

# Final check for any remaining words
if current_words_in_segment:
start_time = current_words_in_segment[0].start
end_time = current_words_in_segment[-1].end + 0.02
if (end_time - start_time) < min_subtitle_duration:
end_time = start_time + min_subtitle_duration

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
srt_content_blocks.append({
"id": subtitle_idx,
"start": start_time,
"end": end_time,
"text": final_subtitle_text
})

return srt_content_blocks # Return list of dicts for easier translation

def assemble_srt_text(srt_blocks):
"""Assembles SRT text from a list of subtitle blocks."""
srt_output = []
for block in srt_blocks:
srt_output.append(str(block["id"]))
srt_output.append(f"{hhmmss_ms(block['start'])} --> {hhmmss_ms(block['end'])}")
srt_output.append(block["text"])
srt_output.append("")
return "\n".join(srt_output)


# New Translation Function using OpenAI API
@st.cache_data(show_spinner=False) # Cache translated results
def translate_text_openai(text_to_translate: str, target_language: str = "English", source_language: str = None) -> str:
"""
   Translates text using OpenAI's Chat Completion API.
   """
if not st.secrets.get("openai_api_key"):
st.warning("OpenAI API key not found in Streamlit secrets. Translation will be skipped.")
return text_to_translate

try:
client = openai.OpenAI(api_key=st.secrets["openai_api_key"])

system_prompt = f"You are a highly accurate translator. Translate the given text into {target_language}."
if source_language:
system_prompt += f" The original language is {source_language}."

response = client.chat.completions.create(
model="gpt-3.5-turbo", # You can try "gpt-4" if you have access and need higher quality (higher cost)
messages=[
{"role": "system", "content": system_prompt},
{"role": "user", "content": text_to_translate}
],
temperature=0.7, # Lower for more literal translation, higher for more creative
max_tokens=500 # Adjust based on expected output length
)
translated_content = response.choices[0].message.content.strip()
return translated_content
except openai.APICallError as e:
st.error(f"OpenAI API Error: {e.response.status_code} - {e.response.json()['error']['message']}")
return f"[Translation Failed: {e.response.json()['error']['message']}] {text_to_translate}"
except Exception as e:
st.error(f"An unexpected error occurred during OpenAI translation: {e}")
return f"[Translation Error] {text_to_translate}"

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
   2. Transcribe the audio using the Faster-Whisper model with word-level timestamps (in its original language).
   3. **(Optional) Translate the transcribed text to English using OpenAI API** (requires API key).
   4. Generate an SRT subtitle file, intelligently segmenting lines based on punctuation, pauses, and configurable duration/character limits.
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

# Translation Toggle and Options
st.sidebar.markdown("---")
st.sidebar.header("Translation Options (OpenAI API)")
enable_translation = st.sidebar.checkbox("Enable Translation to English", value=False,
help="If checked, the transcribed text will be translated to English using OpenAI's API. Requires an OpenAI API key in Streamlit secrets.")

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
words, detected_lang = transcribe_words(_model=model, audio_path=extracted_audio_path, lang=None) 
if not words:
st.error("No words were detected. Please try a clearer audio or a different model size.")
st.stop()

with st.spinner("Building SRT with intelligent segmentation..."):
srt_blocks = bucket_words_by_duration(
words, 
bucket_seconds=bucket_seconds, 
max_chars_per_sub=max_chars, 
min_subtitle_duration=min_duration_seconds
)

final_srt_text = ""
download_filename = os.path.splitext(uploaded_file.name)[0]

if enable_translation:
st.info(f"Translating subtitles from {detected_lang.upper()} to English using OpenAI API...")
translated_srt_blocks = []

# OpenAI API calls can be slow, especially for many small requests.
# It's better to batch them if possible, but for simplicity, we translate block by block here.
# Consider a progress bar here for long videos.
progress_text = "Translation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for i, block in enumerate(srt_blocks):
translated_text = translate_text_openai(block["text"], target_language="English", source_language=detected_lang)
translated_block = block.copy()
translated_block["text"] = translated_text
translated_srt_blocks.append(translated_block)
my_bar.progress((i + 1) / len(srt_blocks), text=progress_text)

my_bar.empty() # Remove progress bar when done
final_srt_text = assemble_srt_text(translated_srt_blocks)
download_filename += "_english_sub.srt"
st.success("Translation complete!")
else:
final_srt_text = assemble_srt_text(srt_blocks)
download_filename += f"_{detected_lang}_sub.srt"

st.success("Done!")
st.subheader("Preview (SRT)")
st.text_area("SRT Content", final_srt_text, height=320)

st.download_button(
"Download SRT",
data=final_srt_text.encode("utf-8"),
file_name=download_filename,
mime="text/plain"
)
