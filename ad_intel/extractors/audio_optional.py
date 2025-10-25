from __future__ import annotations
from pathlib import Path


def extract_audio_features(path: Path) -> dict:
    try:
        from moviepy.editor import VideoFileClip  # type: ignore
        import numpy as np
        import librosa  # type: ignore

        clip = VideoFileClip(str(path))
        if clip.audio is None:
            return {'audio_loudness': 0.0, 'audio_tempo_bpm': 0.0}
        # Extract audio to array (mono)
        audio = clip.audio.to_soundarray(fps=22050)
        clip.close()
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        y = audio.astype('float32')
        sr = 22050
        # Loudness proxy
        rms = float((y**2).mean() ** 0.5)
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return {'audio_loudness': rms, 'audio_tempo_bpm': float(tempo)}
    except Exception:
        # Optional deps missing
        return {}
