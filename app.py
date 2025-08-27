import streamlit as st
import io
import os
import re
from pydub import AudioSegment
import openai
from faster_whisper import WhisperModel
import srt
import datetime

# Setting OpenAI API key
if "openai_api_key" not in st.secrets:
    st.error("OpenAI API key is missing. Please add it to Streamlit secrets.")
    st.stop()

openai.api_key = st.secrets["openai_api_key"]

@st.cache_resource
def load_whisper_model(model_size: str):
    """
    Loads the Whisper model from cache.
    Args:
        model_size (str): The size of the model to load (e.g., "small", "medium").
    Returns:
        WhisperModel: The loaded Whisper model.
    """
    # Use CPU since we are not using a GPU instance.
    return WhisperModel(model_size, device="cpu", compute_type="int8_float16")

@st.cache_resource
def transcribe_words(_model: WhisperModel, audio_path: str, lang: str):
    """
    Transcribes audio to text and provides word-level timestamps.
    Args:
        _model (WhisperModel): The Whisper model.
        audio_path (str): Path to the audio file.
        lang (str): Language code (e.g., "zh").
    Returns:
        List of dicts: List of transcribed words with timestamps and text.
    """
    segments, _ = _model.transcribe(
        audio_path,
        word_timestamps=True,
        language=lang,
        beam_size=5,
        vad_filter=True
    )
    words = []
    for segment in segments:
        for word in segment.words:
            words.append(word)
    return words

def format_time(seconds: float) -> str:
    """
    Formats seconds into SRT timestamp format.
    Args:
        seconds (float): The time in seconds.
    Returns:
        str: Formatted time string (HH:MM:SS,mmm).
    """
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds * 1000):03}"

def bucket_words_by_duration(
    words: list,
    bucket_seconds: int = 5,
    max_chars_per_sub: int = 60
) -> list:
    """
    Buckets words into subtitle segments based on duration and max characters,
    with an intelligent punctuation-based segmentation.

    Args:
        words (list): List of word objects from faster-whisper.
        bucket_seconds (int): Maximum duration for a subtitle block in seconds.
        max_chars_per_sub (int): Maximum number of characters per subtitle line.

    Returns:
        list: A list of dictionaries, each representing an SRT block.
    """
    # Initialize subtitle_idx to 1
    subtitle_idx = 1
    
    srt_content_blocks = []
    current_segment_start_time = 0.0
    current_segment_words = []
    
    def is_sentence_ending(word_text: str) -> bool:
        """
        Checks if a word ends with punctuation that marks the end of a sentence or a pause.
        """
        return word_text.endswith(('.', '?', '!', ',', '。', '？', '！', '，'))

    for i, word in enumerate(words):
        # Use word.word for newer faster-whisper versions
        word_text = word.word
        current_segment_words.append(word)

        if not current_segment_words:
            current_segment_start_time = word.start

        current_duration = word.end - current_segment_start_time
        current_text = "".join([w.word for w in current_segment_words])

        # Check for sentence end, max duration, or max characters
        is_end_of_sentence = is_sentence_ending(word_text)
        is_max_duration = current_duration >= bucket_seconds
        is_max_chars = len(current_text) >= max_chars_per_sub
        
        # Determine if we should create a new subtitle block
        if (is_end_of_sentence and len(current_text) > 5) or is_max_duration or is_max_chars or (i == len(words) - 1):
            if current_segment_words:
                start_time = current_segment_words[0].start
                end_time = current_segment_words[-1].end
                subtitle_text = "".join([w.word for w in current_segment_words])

                srt_content_blocks.append({
                    "id": subtitle_idx,
                    "start": start_time,
                    "end": end_time,
                    "text": subtitle_text
                })
                subtitle_idx += 1
                current_segment_words = []
                current_segment_start_time = 0.0

    return srt_content_blocks


def translate_text_openai(text_content: str, source_lang: str = "zh", target_lang: str = "my") -> str:
    """
    Translates text content using the OpenAI Chat API.

    Args:
        text_content (str): The text to translate.
        source_lang (str): The source language.
        target_lang (str): The target language.
    Returns:
        str: The translated text, or an error message if translation fails.
    """
    if not text_content:
        return ""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional subtitle translator. Translate the following text from {source_lang} to {target_lang}."
                },
                {
                    "role": "user",
                    "content": text_content
                }
            ],
            temperature=0.3,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        translated_content = response.choices[0].message.content.strip()
        return translated_content
    except openai.APIError as e: # CHANGED: Use a more general APIError for backwards compatibility
        st.error(f"OpenAI API Error: {e.response.status_code} - {e.response.json().get('error', {}).get('message', 'Unknown error')}")
        return f"[ဘာသာပြန်မအောင်မြင်ပါ: {e.response.json().get('error', {}).get('message', 'Unknown error')}]"
    except Exception as e:
        st.error(f"An unexpected error occurred during translation: {e}")
        return f"[ဘာသာပြန်မအောင်မြင်ပါ: {e}]"


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(
        page_title="Audio Subtitle Generator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("MP3 Subtitle Generator")
    st.markdown("အသံဖိုင်မှ စာတန်းထိုးများကို ဖန်တီးပါ။")

    st.sidebar.header("Options")
    model_size = st.sidebar.selectbox(
        "Select Whisper model size",
        ("tiny", "base", "small", "medium"),
        index=2
    )

    bucket_seconds = st.sidebar.slider(
        "Subtitle Segment Duration (seconds)",
        min_value=1,
        max_value=10,
        value=5
    )

    max_chars = st.sidebar.slider(
        "Max Characters Per Subtitle Line",
        min_value=30,
        max_value=120,
        value=60
    )

    uploaded_file = st.file_uploader("Upload an MP3 Audio File", type=["mp3"])

    if uploaded_file:
        audio_bytes = uploaded_file.read()
        audio_io = io.BytesIO(audio_bytes)

        st.audio(audio_io, format='audio/mp3')

        with st.spinner("Converting MP3 to WAV..."):
            audio_segment = AudioSegment.from_mp3(audio_io)
            wav_path = "temp_audio.wav"
            audio_segment.export(wav_path, format="wav")

        st.success("Conversion successful! Now transcribing...")

        model = load_whisper_model(model_size)

        with st.spinner("Transcribing with word-level timestamps..."):
            words = transcribe_words(_model=model, audio_path=wav_path, lang="zh")

        os.remove(wav_path)

        if not words:
            st.error("Transcription failed. No words found in the audio.")
            st.stop()

        with st.spinner("Building SRT with intelligent segmentation..."):
            srt_blocks = bucket_words_by_duration(
                words,
                bucket_seconds=bucket_seconds,
                max_chars_per_sub=max_chars,
            )

        if not srt_blocks:
            st.error("Failed to generate SRT subtitles. No words could be segmented.")
            st.stop()

        srt_content = srt.compose(
            (srt.Subtitle(
                index=block['id'],
                start=datetime.timedelta(seconds=block['start']),
                end=datetime.timedelta(seconds=block['end']),
                content=block['text']
            ) for block in srt_blocks),
            reindex=False
        )

        st.subheader("Generated Subtitles (Original Language)")
        st.download_button(
            label="Download SRT File (Original)",
            data=srt_content,
            file_name="subtitles_original.srt",
            mime="text/plain",
        )

        st.code(srt_content, language="srt")
        
        st.markdown("---")
        st.header("Translate to English?")
        if st.checkbox("Enable English Translation"):
            with st.spinner("Translating to English..."):
                translated_blocks = []
                for block in srt_blocks:
                    translated_text = translate_text_openai(block['text'], source_lang="zh", target_lang="en")
                    translated_blocks.append({
                        "id": block['id'],
                        "start": block['start'],
                        "end": block['end'],
                        "text": translated_text
                    })

                translated_srt_content = srt.compose(
                    (srt.Subtitle(
                        index=block['id'],
                        start=datetime.timedelta(seconds=block['start']),
                        end=datetime.timedelta(seconds=block['end']),
                        content=block['text']
                    ) for block in translated_blocks),
                    reindex=False
                )

                st.subheader("Translated Subtitles (English)")
                st.download_button(
                    label="Download SRT File (English)",
                    data=translated_srt_content,
                    file_name="subtitles_translated.srt",
                    mime="text/plain",
                )
                st.code(translated_srt_content, language="srt")

if __name__ == "__main__":
    main()
