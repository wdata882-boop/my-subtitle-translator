# -*- coding: utf-8 -*-
import os
import math
import tempfile
import subprocess
import streamlit as st
import openai # pip install openai

# Faster-Whisper ကို word-level timestamps အတွက် အသုံးပြုရန်
# pip install faster-whisper
from faster_whisper import WhisperModel

# pydub ကို audio ထုတ်ယူရန် အသုံးပြုရန်
# pip install pydub
from pydub import AudioSegment

# ----------------------------
# ကိန်းသေများ (Constants) နှင့် UI Configuration
# ----------------------------
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL_SIZE = "base" # အရည်အသွေး ပိုကောင်းစေရန် "small" (သို့) "medium" ကို ထည့်သွင်းစဉ်းစားနိုင်သည်
MAX_CHARS_PER_SUB = 60 # စာတန်းထိုးတစ်ကြောင်းလျှင် အများဆုံး စာလုံးရေ
DEFAULT_BUCKET_SECONDS = 5 # စာတန်းထိုးအစုအဝေးများအတွက် မူရင်းကြာချိန် (အများဆုံးကြာချိန်)
DEFAULT_MIN_SUBTITLE_DURATION = 0.5 # စာတန်းထိုးတစ်ခုအတွက် အနည်းဆုံးပြသရမည့်ကြာချိန်

# ----------------------------
# ကူညီပေးသည့် Functions များ (Helper Functions)
# ----------------------------
def hhmmss_ms(seconds: float) -> str:
    """
    စက္ကန့်ကို SRT အချိန် Format HH:MM:SS,mmm သို့ ပြောင်းလဲပေးသည်။
    """
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Faster-Whisper Model ကို ကောင်းစွာ cache လုပ်နိုင်ရန် hash_funcs ထည့်သွင်းထားသည်
@st.cache_resource(hash_funcs={WhisperModel: lambda _: None})
def load_model(model_size: str):
    """
    Faster-Whisper model ကို load လုပ်သည်။ စွမ်းဆောင်ရည်အတွက် cache ကို အသုံးပြုသည်။
    """
    st.info(f"Faster-Whisper model ကို load လုပ်နေသည်- {model_size}။ Model ကြီးလျှင် အချိန်ကြာနိုင်သည်။")
    # Streamlit Cloud CPU တွင် ValueError မဖြစ်စေရန် compute_type="int8" ကို အများအားဖြင့် အကောင်းဆုံးအဖြစ် သတ်မှတ်သည်။
    return WhisperModel(model_size, device="cpu", compute_type="int8")

# transcribe_words function မှာ `lang` parameter ကို လက်ခံရန် ပြင်ဆင်ထားသည်
@st.cache_resource(hash_funcs={WhisperModel: lambda _: None}) # WhisperModel ကို hashing မှ ချန်လှပ်ထားသည်
def transcribe_words(_model: WhisperModel, audio_path: str, lang: str = None):
    """
    Faster-Whisper ကို အသုံးပြု၍ အသံကို စကားလုံးအဆင့် အချိန်မှတ်တမ်းများဖြင့် transcribe လုပ်သည်။
    lang=None ဆိုလျှင် ဘာသာစကားကို အလိုအလျောက် သိရှိပြီး ထိုဘာသာစကားဖြင့် transcribe လုပ်သည်။
    """
    st.info(f"အသံကို စကားလုံးအဆင့် အချိန်မှတ်တမ်းများဖြင့် transcribe လုပ်နေသည်။ ဘာသာစကား- {lang if lang else 'အလိုအလျောက် သိရှိမည်'}")
    segments, info = _model.transcribe(
        audio_path,
        word_timestamps=True, 
        language=lang, # ဤနေရာတွင် ပေးပို့လာသော lang parameter ကို အသုံးပြုမည်။
        beam_size=7 
    )
    words = []
    for segment in segments:
        for word in segment.words:
            words.append(word)
    
    detected_language = info.language
    st.info(f"သိရှိရသော ဘာသာစကား- {detected_language.upper()}")
    return words, detected_language

# ပြင်ဆင်ထားသည်- `word.text` ကို `word.word` အဖြစ် ပြောင်းလဲထားသည် (`bucket_words_by_duration` တွင်)
def bucket_words_by_duration(words: list, bucket_seconds: int = 5, max_chars_per_sub: int = 60, min_subtitle_duration: float = 0.5) -> list:
    """
    သဘာဝအားဖြင့် ရပ်တန့်ခြင်း (ပုဒ်ဖြတ်ပုဒ်ရပ်များ) အပေါ် အခြေခံ၍ စာတန်းထိုး block များကို (dict များ၏ list) ဖန်တီးပြီး၊
    လိုအပ်ပါက ကြာချိန်/စာလုံးရေ ကန့်သတ်ချက်များအတိုင်း ခွဲခြမ်းသည်။
    စာတန်းထိုးတိုင်းသည် အနည်းဆုံးပြသရမည့်ကြာချိန်နှင့် ကိုက်ညီကြောင်း သေချာစေသည်။
    """
    # ဤနေရာတွင် subtitle_idx ကို အစောပိုင်း အစပျိုးပေးသည်
    srt_content_blocks = [] # စာတန်းထိုး block တစ်ခုစီအတွက် dict များ၏ list ကို သိမ်းဆည်းရန်
    subtitle_idx = 1
    current_words_in_segment = []

    def ends_with_punctuation(word_text):
        # အင်္ဂလိပ်နှင့် တရုတ်ပုဒ်ဖြတ်ပုဒ်ရပ်များကို ကိုင်တွယ်သည်။
        return word_text.endswith(('.', '?', '!', ',', '。', '？', '！', '，'))

    for i, word in enumerate(words):
        # ပြင်ဆင်ထားသည့်လိုင်း- faster-whisper ၏ WordInfo object structure အတွက် word.word ကို အသုံးပြုပါ။
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
                if (next_word_start - word.end) > 0.4: # သိသာထင်ရှားသော ရပ်တန့်မှု (ဥပမာ: 0.4 စက္ကန့်)
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
                end_time = current_words_in_segment[-1].end + 0.02 # ချောမွေ့သော ပြောင်းလဲမှုအတွက် 20ms offset ထပ်ထည့်သည်။
                
                if (end_time - start_time) < min_subtitle_duration:
                    end_time = start_time + min_subtitle_duration

                subtitle_text = " ".join([w.word.strip() for w in current_words_in_segment]).strip()
                
                # ဖတ်ရလွယ်ကူစေရန် စာကြောင်းခွဲခြင်း
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
                
                current_words_in_segment = [] # နောက် segment အတွက် ပြန်လည်သတ်မှတ်သည်။

    # ကျန်ရှိသော စကားလုံးများအတွက် နောက်ဆုံးစစ်ဆေးမှု
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

    return srt_content_blocks # ဘာသာပြန်ရန် ပိုမိုလွယ်ကူစေရန် dict များ၏ list ကို ပြန်ပေးသည်။

def assemble_srt_text(srt_blocks):
    """Subtitle block များ၏ list မှ SRT text ကို ပေါင်းစည်းသည်။"""
    srt_output = []
    for block in srt_blocks:
        srt_output.append(str(block["id"]))
        srt_output.append(f"{hhmmss_ms(block['start'])} --> {hhmmss_ms(block['end'])}")
        srt_output.append(block["text"])
        srt_output.append("")
    return "\n".join(srt_output)


# OpenAI API ကို အသုံးပြုထားသော Translation Function အသစ်
@st.cache_data(show_spinner=False) # ဘာသာပြန်ထားသော ရလဒ်များကို cache လုပ်သည်။
def translate_text_openai(text_to_translate: str, target_language: str = "English", source_language: str = None) -> str:
    """
    OpenAI ၏ Chat Completion API ကို အသုံးပြု၍ စာသားကို ဘာသာပြန်သည်။
    """
    if not st.secrets.get("openai_api_key"):
        st.warning("Streamlit secrets တွင် OpenAI API key မတွေ့ပါ။ ဘာသာပြန်ခြင်းကို ကျော်သွားပါမည်။")
        return text_to_translate

    try:
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        
        system_prompt = f"သင်သည် အလွန်တိကျသော ဘာသာပြန်သူတစ်ဦးဖြစ်သည်။ ပေးထားသော စာသားကို {target_language} သို့ ဘာသာပြန်ပါ။"
        if source_language:
            system_prompt += f" မူရင်းဘာသာစကားမှာ {source_language} ဖြစ်သည်။"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # ပိုမိုကောင်းမွန်သော အရည်အသွေး လိုအပ်ပါက "gpt-4" ကို စမ်းသပ်နိုင်သည် (ကုန်ကျစရိတ်ပိုများသည်)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_translate}
            ],
            temperature=0.7, # တိကျသော ဘာသာပြန်အတွက် လျှော့ချပါ၊ ပိုမိုဖန်တီးမှုရှိသော ဘာသာပြန်အတွက် တိုးမြှင့်ပါ။
            max_tokens=500 # မျှော်လင့်ထားသော output အရှည်ပေါ်မူတည်၍ ချိန်ညှိပါ။
        )
        translated_content = response.choices[0].message.content.strip()
        return translated_content
    except openai.APICallError as e:
        st.error(f"OpenAI API Error: {e.response.status_code} - {e.response.json().get('error', {}).get('message', 'အမည်မသိ အမှား')}")
        return f"[ဘာသာပြန်မအောင်မြင်ပါ: {e.response.json().get('error', {}).get('message', 'အမည်မသိ အမှား')}] {text_to_translate}"
    except Exception as e:
        st.error(f"OpenAI ဘာသာပြန်နေစဉ် မမျှော်လင့်ထားသော အမှားတစ်ခု ဖြစ်ပေါ်ခဲ့သည်: {e}")
        return f"[ဘာသာပြန် အမှား] {text_to_translate}"

def ensure_ffmpeg_access():
    """
    ffmpeg executable ကို system PATH တွင် ရှိနေကြောင်း သေချာစေသည်။
    ၎င်းကို deployment အတွင်း packages.txt မှတဆင့် install လုပ်လိမ့်မည်။
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except FileNotFoundError:
        st.error(f"FFmpeg executable မတွေ့ပါ။ 'packages.txt' မှတဆင့် install လုပ်သင့်သည်။ သင်၏ packages.txt ဖိုင်ကို စစ်ဆေးပြီး 'ffmpeg' ပါဝင်ကြောင်း သေချာပါစေ။")
        return False
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg version စစ်ဆေးနေစဉ် အမှားပြန်ပေးသည်: {e}။ ၎င်းသည် packages.txt မှတဆင့် ffmpeg install လုပ်ရာတွင် ပြဿနာရှိနိုင်ကြောင်း ပြသသည်။")
        return False
    except Exception as e:
        st.error(f"FFmpeg setup လုပ်နေစဉ် မမျှော်လင့်ထားသော အမှားတစ်ခု ဖြစ်ပေါ်ခဲ့သည်: {e}။ ကျေးဇူးပြု၍ deployment logs များကို အသေးစိတ်ကြည့်ပါ။")
        return False

def extract_audio_pydub(input_path: str, output_path: str, sr: int = 16000) -> str:
    """
    pydub ကို အသုံးပြု၍ mono wav @16kHz ကို ထုတ်ယူသည်။
    Pydub သည် system PATH တွင် 'ffmpeg' ကို အလိုအလျောက် ရှာဖွေလိမ့်မည်။
    အောင်မြင်ပါက ထုတ်ယူထားသော audio ၏ path ကို ပြန်ပေးသည်၊ မဟုတ်ပါက None။
    """
    try:
        AudioSegment.converter = "ffmpeg"
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(sr)  # 16kHz
        audio.export(output_path, format="wav")
        st.success(f"အသံကို အောင်မြင်စွာ ထုတ်ယူပြီးသည်- {output_path}")
        return output_path
    except Exception as e:
        st.error(f"pydub ဖြင့် အသံထုတ်ယူရာတွင် အမှားဖြစ်သည် (FFmpeg ပြဿနာ?): {e}။ သင်၏ video ဖိုင် format ကို စစ်ဆေးပြီး 'ffmpeg' ကောင်းစွာ အလုပ်လုပ်ကြောင်း သေချာပါစေ။")
        return None
