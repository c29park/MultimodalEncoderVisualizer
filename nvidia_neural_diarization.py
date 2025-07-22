import torchaudio
import soundfile as sf
import nemo.collections.asr as nemo_asr
import librosa
import json
from dotenv import load_dotenv
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from utils import extract_audio
import os
load_dotenv()
stt_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_conformer_ctc_large")
diarizer = NeuralDiarizer.from_pretrained("diar_msdd_telephonic")
CONFIG_PATH = "conf/diar_infer_telephonic.yaml"
def _init_diarizer(cfg):
    return NeuralDiarizer(cfg=cfg)

def run_diarization(video_path:str) -> list[dict]:
    """
    Performs audio-visual speaker diarization on the video file using NeMo
    Returns list of {'start': float, 'end': float, 'speaker': str}    
    """
    if video_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
        audio_path = extract_audio(video_path)
    else:
        audio_path = video_path

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    manifest = {"audio_filepath": audio_path, "duration": duration}
    manifest_path = "temp.manifest"
    with open(manifest_path, "w") as mf:
        mf.write(json.dumps(manifest))

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir= "diar_output"

    diarizer = _init_diarizer(cfg)
    diarizer.diarize()

    rttm_file = os.path.join(cfg.diarizer.out_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".rttm")
    segments = []
    with open(rttm_file, "r") as rf:
        for line in rf:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append({
                "start": start,
                "end": start + duration,
                "speaker": speaker
            })
    return segments
        

def transcribe_segments(audio_path:str, segments: list[dict], target_sr: int=16000) -> list[dict]:
    """
    Transcribes each diarization segment using NeMo ASR
    Returns {'speaker', 'start', 'end', 'text'}
    """

    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr
    
    results = []
    for seg in segments:
        s, e = seg['start'], seg['end']
        start_frame = int(s * sr)
        end_frame = int(e * sr)
        snippet = waveform[:, start_frame:end_frame]
        temp_path = f"temp_{int(s*1000)}_{int(e*1000)}.wav"
        sf.write(temp_path, snippet.numpy().T,sr)
        text = stt_model.transcribe([temp_path])[0]
        os.remove(temp_path)
        results.append({
            "speaker": seg['speaker'],
            "start": s,
            "end": e,
            "text": text
        })
    return results