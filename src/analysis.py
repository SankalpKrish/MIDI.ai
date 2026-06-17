from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import librosa

logger = logging.getLogger(__name__)

class GlobalAnalyzer:
    def analyze(self, key_source_path: str | Path | None, bpm_source_path: str | Path | None = None) -> dict[str, float | str]:
        key_source_name = Path(key_source_path).name if key_source_path else "None"
        bpm_source_name = Path(bpm_source_path if bpm_source_path else key_source_path).name if (bpm_source_path or key_source_path) else "None"
        logger.info("Performing global analysis (Key: %s, BPM Source: %s)...", key_source_name, bpm_source_name)
        
        # Load audio for Key/Tuning
        y_key, sr_key = librosa.load(key_source_path, sr=None)
        
        # Load audio for BPM (if different)
        if bpm_source_path and bpm_source_path != key_source_path:
            y_bpm, sr_bpm = librosa.load(bpm_source_path, sr=None)
        else:
            y_bpm, sr_bpm = y_key, sr_key

        # 1. Tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=y_bpm, sr=sr_bpm)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_bpm)
        bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

        # 2. Key/Tonality
        key = self._estimate_key(y_key, sr_key)

        # 3. Tuning Standard
        tuning_hz = self._estimate_tuning_standard(y_key, sr_key)

        return {
            "bpm": round(bpm, 2),
            "key": key,
            "tuning_hz": round(tuning_hz, 2)
        }

    def _estimate_key(self, y: np.ndarray, sr: int) -> str:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)

        # Krumhansl-Schmuckler profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        major_profile /= np.linalg.norm(major_profile)
        minor_profile /= np.linalg.norm(minor_profile)
        chroma_avg /= np.linalg.norm(chroma_avg)

        major_corrs = []
        minor_corrs = []

        for i in range(12):
            major_corrs.append(np.dot(chroma_avg, np.roll(major_profile, i)))
            minor_corrs.append(np.dot(chroma_avg, np.roll(minor_profile, i)))

        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        best_major_idx = np.argmax(major_corrs)
        best_minor_idx = np.argmax(minor_corrs)

        if major_corrs[best_major_idx] > minor_corrs[best_minor_idx]:
            return f"{pitches[best_major_idx]} Major"
        else:
            return f"{pitches[best_minor_idx]} Minor"

    def _estimate_tuning_standard(self, y: np.ndarray, sr: int) -> float:
        # Focus on A4 range (400-500Hz)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=400, fmax=500)
        
        valid_mask = pitches > 0
        valid_pitches = pitches[valid_mask]
        valid_mags = magnitudes[valid_mask]
        
        if len(valid_pitches) == 0:
            return 440.0
            
        a4_mask = (valid_pitches >= 420) & (valid_pitches <= 460)
        
        if not np.any(a4_mask):
             return 440.0

        a4_pitches = valid_pitches[a4_mask]
        a4_mags = valid_mags[a4_mask]
        
        tuning = np.average(a4_pitches, weights=a4_mags)
        return tuning
