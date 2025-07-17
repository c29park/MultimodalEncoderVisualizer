import logging
import re
# suppress Streamlit’s “missing ScriptRunContext” warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
import os
import streamlit as st
from utils import extract_audio
from pyannote_diarization import diarize, transcribe_segments
# Basic Streamlit app for multimodal analysis of YouTube videos with modality toggles

def main():
    st.set_page_config(page_title="Multimodal Video Analyzer", layout="wide")
    st.title("Multimodal Video Analyzer")
    # st.markdown(
    #     """
    #     Enter the YouTube video link below to analyze pose, facial expressions, and vocal cues.
    #     Use the checkboxes to enable or disable specific modalities before running the analysis.
    #     The app will detect changes in selected modalities and align them with speaker diarization timestamps.
    #     """
    # )
    st.markdown(
    """
    Upload a video file to analyze multimodal features like pose, facial expressions, and vocal cues tied with speaker segments and transcriptions.
    """
    )

    # Input for YouTube video URL
    # youtube_url = st.text_input(
    #     label="YouTube Video URL",
    #     placeholder="https://www.youtube.com/watch?v=..."
    # )
    uploaded_video = st.file_uploader("Upload Video File", type = ["mp4", "mkv", "mov", "avi", "webm"])
    if not uploaded_video:
        st.info("Please upload a video file to analyze.")
        return

    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getvalue())
    st.success(f"Saved uploaded video to {video_path}")
    # Modality toggles
    st.subheader("Select Modalities to Analyze")
    enable_pose = st.checkbox("Pose Detection", value=True)
    enable_facial = st.checkbox("Facial Expression", value=True)
    enable_vocal = st.checkbox("Vocal Features", value=True)
    # Analyze button
    if st.button("Analyze Video"):
        # if not youtube_url:
        #     st.error("Please enter a valid YouTube video URL.")
        #     return
        # #Validate URL Format
        # url_pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+'
        # if not re.match(url_pattern, youtube_url):
        #     st.error("Invalid YouTube URL. Please enter a valid link.")
        #     return
        # #Download the video
        # st.info(f"Downloading video from {youtube_url}...")
        # try:
        #     video_path = download_video(youtube_url)
        #     st.success(f"Video downloaded successfully.")
        # except Exception as e:
        #     st.error(f"Failed to download video: {e}")
        #     return
        # Extract audio
        st.info("Extracting audio from the video...")
        try:
            audio_file = extract_audio(video_path)
            st.success(f"Audio extracted successfully: {audio_file}")
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")
            return
        #Audio-Visual Diarization
        st.info("Running speaker diarization...")
        try: 
            segments = diarize(audio_file, cache_dir ="diarization_cache")
            st.success("Diarization completed successfully.")
        except Exception as e:
            st.error(f"Speaker diarization failed: {e}")
            return
        #Transcribe segments
        st.info("Transcribing diarization segments...")
        try:
            entries = transcribe_segments(segments, audio_file)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return
        
        # Display diarization and transcription
        st.subheader("Diarization and Transcription Results")
        for e in entries:
            st.write(f"{e['start']:.2f}s - {e['end']:.2f}s [{e['speaker']}]: {e['text']}")

        
        # Report which modalities are active
        active = []
        if enable_pose:
            active.append("Pose")
        if enable_facial:
            active.append("Facial Expression")
        if enable_vocal:
            active.append("Vocal Features")
        st.write(f"**Enabled modalities:** {', '.join(active) if active else 'None'}")

        # Placeholder for processing logic
        with st.spinner("Detecting multimodal changes..."):
            # TODO: Call backend processing functions here and pass enable_pose, enable_facial, enable_vocal
            # e.g. results = analyze_video(youtube_url, enable_pose, enable_facial, enable_vocal)
            pass

        st.success("Analysis complete! See results below.")
        # Placeholder for results output
        st.subheader("Detected Changes")
        # Example outputs - to be replaced with dynamic content
        if enable_facial:
            st.write("- Speaker A (00:15): Facial expression changed from neutral to surprise.")
        if enable_pose:
            st.write("- Speaker B (00:47): Hand pose changed (raised hand).")
        if enable_vocal:
            st.write("- Speaker A (01:10): Speech energy spike detected.")

if __name__ == "__main__":
    main()