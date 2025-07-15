import re
import subprocess

import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

import streamlit as st
import yt_dlp
import os

@st.cache_data(show_spinner=False)
def download_video(url: str, cookie_file: str = None) -> str:
    """
    Downloads the YouTube video from the given URL and returns the local file path. 
    """
    if not cookie_file:
        st.warning("If this video is age/region restricted, please provide a cookie file.")
        uploaded = st.file_uploader("Upload Cookie File", type= ["txt"])
        if uploaded:
            cookie_dir = "cookies"
            os.makedirs(cookie_dir, exist_ok=True)
            cookie_file = os.path.join(cookie_dir, uploaded.name)
            with open(cookie_file, "wb") as f:
                f.write(uploaded.getvalue())
            st.success(f"Saved cookies file at {cookie_file}")

    output_path = "downloads"
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(output_path,"%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "verbose": True, 
    }
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    else:
        ydl_opts["cookies_from_browser"] = "firefox"

    print(">>> yt-dlp options:", ydl_opts)

    #Run download
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    video_id = info.get("id")
    ext = info.get("ext", "mp4")
    filepath = os.path.join(output_path, f"{video_id}.{ext}")
    
    return filepath


def extract_audio(video_path: str, audio_path: str = None) -> str:
    """
    Used for extracting audio( 16kHz mono WAV) from the video using ffmpeg.
    Path to the WAV file is returned. 
    """
    if audio_path is None:
        base, _ = os.path.splitext(video_path)
        audio_path = base + ".wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(cmd, check=True)
    return audio_path
