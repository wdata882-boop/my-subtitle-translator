# Duration-based Subtitle Maker (Streamlit)

Generate SRT subtitles from video/audio using fixed duration per line (not model segments). Powered by Faster-Whisper (word timestamps).

## Features
- Upload video/audio (mp4, mov, mkv, avi, mp3, wav, m4a)
- Extract audio via FFmpeg (mono 16k)
- Transcribe with faster-whisper (word timestamps)
- Build SRT by fixed duration buckets (e.g., 3s/line)
- Control max chars per subtitle
- Download SRT

## Local Run
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py