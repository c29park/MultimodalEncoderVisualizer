import os
import json
import hashlib
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline
import torchaudio
from collections import defaultdict
load_dotenv()
def _get_cache_path(audio_file: str, cache_dir: str) -> str:
    """
    Computes a cache filename for the given audio file. 
    """
    file_hash = hashlib.md5(os.path.abspath(audio_file).encode()).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{file_hash}.json")

def diarize(audio_path:str, model:str = "pyannote/speaker-diarization-3.1", cache_dir: str = "diarization_cache")-> list[dict]:
    """
    Perform speaker diarization on an audio file. Caches resuls to avoid re-computation.

    """
    cache_path = _get_cache_path(audio_path, cache_dir)
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
        
    pipeline=Pipeline.from_pretrained(model, use_auth_token=True)
    annotation = pipeline(audio_path)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3)
        })
    
    with open(cache_path, "w") as f:
        json.dump(segments, f)
    return segments

def transcribe_segments(
    segments: list[dict],
    audio_path: str,
    asr_model: str = "openai/whisper-medium",
    chunk_length_s: float=30.0,
    stride_length_s: float=5.0,
    device: int =0
) -> dict[str, str]:
    """
    Transcribe diarized segments from an audio file. 
    """
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model = asr_model,
        chunk_length_s = chunk_length_s,
        stride_length_s = stride_length_s,
        device=device
    )

    waveform, sr = torchaudio.load(audio_path, backend="ffmpeg")
    results = []

    for seg in segments:
        start_frame = int(seg["start"] * sr)
        end_frame = int(seg["end"] * sr)
        segment_audio = waveform[:, start_frame:end_frame]

        raw_array = segment_audio.mean(dim=0).numpy()

        result = asr({"raw": raw_array, "sampling_rate": sr})
        text = result.get("text", "").strip()
        results.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })
    
    return results

