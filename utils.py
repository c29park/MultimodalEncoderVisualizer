import re
import subprocess

import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

import streamlit as st
import yt_dlp
import os

@st.cache_data(show_spinner=False)
def download_video(url: str)->str:
    """
    Downloads the YouTube video from the given URL and returns the local file path. 
    """
    output_path = "downloads"
    os.makedirs(output_path, exist_ok=True)
    cookie_file = "cookies.txt"
    if not os.path.exists(cookie_file):
        cookies_args = ["--cookies-from-browser", "firefox"]
    else:
        cookies_args = ["--cookies", cookie_file]
    cmd = [
        "yt-dlp",
        "--force-overwrites",
        "-f", "bestvideo[height=240]+bestaudio",
        *cookies_args,
        "--user-agent", 
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
        "-o", f"{output_path}/%(id)s.%(ext)s",
        url
    ]

    # Debug: show the command
    print("Running command:", " ".join(cmd))

    # Execute download
    try:
        # Run the command and capture output for debugging
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Print stdout/stderr to help diagnose
            print("yt-dlp stdout:", result.stdout)
            print("yt-dlp stderr:", result.stderr)
            raise RuntimeError(f"Download failed (exit {result.returncode})")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed with exit code {e.returncode}")


    # Retrieve video ID to locate file
    try:
        vid_id = subprocess.check_output([
            "yt-dlp", "--get-id", url
        ], text=True).strip()
    except subprocess.CalledProcessError:
        vid_id = None

    # Construct path
    if vid_id:
        return os.path.join(output_path, f"{vid_id}.mp4")

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


