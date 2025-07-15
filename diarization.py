import torchaudio
import soundfile as sf
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import SortformerEncLabelModel
from utils import extract_audio
import os

stt_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_conformer_ctc_large")
diarizer = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")

def run_diarization(video_path:str) -> list[dict]:
    """
    Performs audio-visual speaker diarization on the video file using NeMo
    Returns list of {'start': float, 'end': float, 'speaker': str}    
    """
    if video_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
        audio_path = extract_audio(video_path)
    else:
        audio_path = video_path

    result = diarizer.diarize([audio_path])[0]
    segments = []
    for utt in result:
        segments.append({
            "start": utt.start,
            "end": utt.end,
            "speaker": utt.label
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