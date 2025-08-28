# -*- coding: utf-8 -*-
import os
import tempfile
import streamlit as st
import assemblyai as aai

# ----------------------------
# UI Configuration
# ----------------------------
st.set_page_config(layout="wide", page_title="Advanced Subtitle Generator")

st.title("Advanced AI Subtitle Generator (using AssemblyAI) 噫")
st.markdown("""
    Upload a video file. This app uses the **AssemblyAI API** for:
    1.  **Best Accuracy Transcription:** Uses AssemblyAI's best model for top-tier accuracy.
    2.  **Speaker Labels:** Automatically detect and label different speakers.
    3.  **On-Screen Text Recognition (OCR):** Extracts text visible in the video.
    4.  **Perfectly Formatted SRT:** Create a subtitle file with accurate timing and punctuation.
""")

# ----------------------------
# Helper Functions
# ----------------------------
def transcribe_with_assemblyai(file_path: str):
    """
    Transcribes a media file using the latest AssemblyAI SDK features.
    """
    api_key = st.secrets.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        st.error("AssemblyAI API Key is not configured in Streamlit Secrets.")
        st.info("Please add your AssemblyAI API key as a secret named 'ASSEMBLYAI_API_KEY'.")
        return None

    aai.settings.api_key = api_key
    
    # --- START OF CORRECTION ---
    # The new version of the AssemblyAI SDK is simpler.
    # All features are boolean flags inside TranscriptionConfig.
    # We no longer need to specify `speech_model` as it uses the best by default.
    config = aai.TranscriptionConfig(
        punctuate=True,
        format_text=True,
        speaker_labels=True,
        auto_highlights=True,
        extract_text=True  # This is the correct way to enable OCR in the new version
    )
    # --- END OF CORRECTION ---

    transcriber = aai.Transcriber()
    
    with st.spinner("Uploading file and processing with AssemblyAI... This may take a moment."):
        try:
            # Pass the file path and the config object to the transcribe method
            transcript = transcriber.transcribe(file_path, config)
        except Exception as e:
            st.error(f"AssemblyAI processing failed: {e}")
            return None

    # Use TranscriptStatus.error in the new version
    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Transcription failed: {transcript.error}")
        return None
    
    return transcript

# ----------------------------
# Streamlit App UI
# ----------------------------
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1.  Get your free API key from [AssemblyAI](https://www.assemblyai.com/).
    2.  In your Streamlit app's settings, add a new secret.
    3.  Set the secret name to `ASSEMBLYAI_API_KEY` and paste your key as the value.
    4.  Upload a video and enjoy the advanced features!
    """
)

uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, MKV, etc.)", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        transcript = transcribe_with_assemblyai(temp_video_path)

        if transcript:
            st.success("Processing Complete!")
            
            # Dynamically create tabs based on available results
            tab_titles = ["SRT Subtitles", "Full Transcript", "Speakers", "Summary"]
            
            # Check for OCR results (now called content_safety_labels in some contexts, but we check text_extractions)
            if transcript.text_extractions:
                tab_titles.append("On-Screen Text (OCR)")
            
            tabs = st.tabs(tab_titles)

            # Tab 1: SRT Subtitles
            with tabs[0]:
                st.subheader("SRT Subtitle File")
                if transcript.text:
                    srt_content = transcript.export_subtitles_srt()
                    st.text_area("SRT Content", srt_content, height=300)
                    
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    dl_name = f"{base_name}_subtitles.srt"
                    st.download_button(
                        label="Download .SRT File",
                        data=srt_content.encode("utf-8"),
                        file_name=dl_name,
                        mime="text/plain"
                    )
                else:
                    st.info("No spoken words found to generate subtitles.")

            # Tab 2: Full Transcript
            with tabs[1]:
                st.subheader("Full Transcript Text")
                st.text_area("Full Text", transcript.text if transcript.text else "No text transcribed.", height=300)

            # Tab 3: Speakers
            with tabs[2]:
                st.subheader("Transcript by Speaker")
                if transcript.utterances:
                    for utterance in transcript.utterances:
                        st.markdown(f"**Speaker {utterance.speaker}:** {utterance.text}")
                else:
                    st.info("No speaker labels were detected in this audio.")

            # Tab 4: Summary
            with tabs[3]:
                st.subheader("Conversation Summary")
                if transcript.auto_highlights and transcript.auto_highlights.results:
                    for result in transcript.auto_highlights.results:
                        st.markdown(f"- {result.text} (found in {len(result.timestamps)} places)")
                else:
                    st.info("No summary could be generated for this audio.")
            
            # Tab 5: OCR Results (if any)
            if transcript.text_extractions:
                with tabs[-1]: # Always access the last tab for OCR
                    st.subheader("Detected On-Screen Text (OCR)")
                    for ocr_result in transcript.text_extractions:
                        start_ms = ocr_result.timestamp.start
                        end_ms = ocr_result.timestamp.end
                        st.markdown(f"**Time:** `{start_ms//1000}s` to `{end_ms//1000}s`")
                        st.info(ocr_result.text)
