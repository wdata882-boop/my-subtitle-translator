# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
import moviepy.editor as mp
from openai import OpenAI
from googletrans import Translator
import pytesseract
from PIL import Image

# ----------------------------
# UI Configuration (Burmese)
# ----------------------------
st.set_page_config(
    page_title="Universal Subtitle Generator",
    page_icon="🎬",
    layout="wide"
)

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
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def get_api_key():
    """
    Retrieves the OpenAI API key from Streamlit secrets or user input.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    st.warning("Please enter your OpenAI API Key to use this app.")
    api_key_input = st.text_input(
        "OpenAI API Key (မရှိပါက ရိုက်ထည့်ပါ)", 
        type="password", 
        help="သင့်ရဲ့ API key ကို Streamlit Cloud ရဲ့ Secrets မှာ ထည့်ထားနိုင်ပါတယ်၊ သို့မဟုတ် ဒီနေရာမှာ တိုက်ရိုက်ရိုက်ထည့်နိုင်ပါတယ်။"
    )
    return api_key_input

def transcribe_with_whisper(audio_path: str, client: OpenAI):
    """
    Transcribes the audio file using OpenAI Whisper API.
    Returns the transcription and word-level timestamps.
    """
    st.info("အသံကို စာသားပြောင်းနေပါသည်... စောင့်ဆိုင်းပေးပါ...")
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="verbose_json",
                language="en" # Setting language to default
            )
        return transcript
    except Exception as e:
        st.error(f"Whisper API ခေါ်ယူမှု မအောင်မြင်ပါ: {e}")
        return None

def translate_to_english(text: str):
    """
    Translates a given text to English using Google Translate.
    """
    st.info("စာသားကို အင်္ဂလိပ်လို ပြန်ဆိုနေပါသည်...")
    try:
        translator = Translator()
        translated_text = translator.translate(text, dest='en').text
        return translated_text
    except Exception as e:
        st.error(f"ဘာသာပြန်ဆိုမှု မအောင်မြင်ပါ: {e}")
        return None

def extract_ocr_text(video_path: str, interval: int = 5):
    """
    Extracts text from video frames using OCR.
    """
    st.info("ဗီဒီယိုထဲက စာသားတွေကို ရှာဖွေဖတ်နေပါသည်...")
    ocr_results = []
    try:
        clip = mp.VideoFileClip(video_path)
        duration = clip.duration
        
        for t in range(0, int(duration), interval):
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            text = pytesseract.image_to_string(img)
            if text.strip():
                ocr_results.append({
                    "start": t,
                    "end": t + interval,
                    "text": f"[On-Screen Text]: {text.strip()}"
                })
        return ocr_results
    except Exception as e:
        st.error(f"OCR လုပ်ဆောင်မှု မအောင်မြင်ပါ: {e}")
        return []

def main():
    """
    Main function for the Streamlit app.
    """
    st.title("🎬 ဗီဒီယိုဖိုင်မှ စာသားနှင့် Subtitle ထုတ်လုပ်ခြင်း")
    st.markdown("""
    ဒီ application က ဗီဒီယိုဖိုင်တွေကနေ အသံနဲ့ စာသားတွေကို တိကျမှန်ကန်စွာ ထုတ်လုပ်ပေးနိုင်ပါတယ်။
    
    **အဓိက လုပ်ဆောင်ချက်များ:**
    * **အမြင့်ဆုံး တိကျမှု:** OpenAI Whisper API ကိုသုံးပြီး အသံကို စာသားပြောင်းလဲပေးခြင်း။
    * **ဘာသာစကားမျိုးစုံ သိရှိခြင်း:** မည်သည့်ဘာသာစကားမဆို အလိုအလျောက် သိရှိပေးခြင်း။
    * **OCR:** ဗီဒီယိုထဲက ပေါ်နေတဲ့ စာသားတွေကို ဖတ်ရှုပေးခြင်း။
    """)
    
    api_key = get_api_key()

    if not api_key:
        return

    st.subheader("အဆင့် ၁: ဗီဒီယိုဖိုင် တင်ပါ")
    uploaded_file = st.file_uploader(
        "ဗီဒီယိုဖိုင် (ဥပမာ- mp4, mov) ကို ရွေးချယ်ပါ", 
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file:
        st.subheader("အဆင့် ၂: ရွေးချယ်စရာများကို သတ်မှတ်ပါ")
        
        translate_to_en = st.checkbox(
            "📝 စာသားကို အင်္ဂလိပ်လို ဘာသာပြန်လိုပါသလား?"
        )
        
        extract_ocr = st.checkbox(
            "📄 ဗီဒီယိုပေါ်က စာသားတွေကိုပါ ထုတ်ယူလိုပါသလား?"
        )
        
        st.write("---")
        
        if st.button("▶️ စတင်ပါ"):
            if not api_key:
                st.error("API Key မရှိပါ၊ ကျေးဇူးပြု၍ ထည့်သွင်းပေးပါ။")
                return

            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    video_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    audio_path = os.path.join(tmpdir, "audio.mp3")
                    
                    # Use moviepy to extract audio
                    st.info("အသံကို ထုတ်ယူနေပါသည်...")
                    video_clip = mp.VideoFileClip(video_path)
                    video_clip.audio.write_audiofile(audio_path, logger=None)

                    # Initialize OpenAI client
                    client = OpenAI(api_key=api_key)
                    
                    # Transcribe
                    transcript_data = transcribe_with_whisper(audio_path, client)
                    if not transcript_data:
                        return
                    
                    # Translate if selected
                    if translate_to_en:
                        translated_text = translate_to_english(transcript_data.text)
                        if translated_text:
                            transcript_data.text = translated_text
                    
                    # OCR if selected
                    ocr_results = []
                    if extract_ocr:
                        ocr_results = extract_ocr_text(video_path)
                    
                    # Generate SRT
                    st.subheader("ရလာဒ်")
                    
                    srt_content = ""
                    segment_id = 1
                    
                    # Add OCR results at the beginning of the file to combine them.
                    all_segments = list(transcript_data.segments)
                    all_segments.extend(ocr_results)

                    # Sort segments by start time
                    all_segments.sort(key=lambda s: s['start'])

                    for segment in all_segments:
                        start_time = segment.get('start', 0)
                        end_time = segment.get('end', 0)
                        text = segment.get('text', "")
                        
                        if end_time > start_time:
                            srt_content += f"{segment_id}\n"
                            srt_content += f"{hhmmss_ms(start_time)} --> {hhmmss_ms(end_time)}\n"
                            srt_content += f"{text.strip()}\n\n"
                            segment_id += 1
                        
                    if srt_content:
                        st.text_area("SRT Content", srt_content, height=320)
                        
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        dl_name = f"{base_name}_english_sub.srt" if translate_to_en else f"{base_name}_sub.srt"
                        
                        st.download_button(
                            "SRT ဖိုင်ကို ဒေါင်းလုဒ်လုပ်ရန်",
                            srt_content,
                            file_name=dl_name,
                            mime="text/plain"
                        )
                    else:
                        st.warning("စာသားတစ်ခုမှ မတွေ့ပါ")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

if __name__ == "__main__":
    main()
