from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    output_dir: str | Path = "output"
    demucs_model: str = "htdemucs"
    demucs_shifts: int = 1
    demucs_overlap: float = 0.25
    quantize_division: int = 4
    ghost_note_threshold: float = 0.1
    yamnet_top_n: int = 10
    yamnet_url: str = "https://tfhub.dev/google/yamnet/1"
    yamnet_class_url: str = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    key_fmin: int = 400
    key_fmax: int = 500
    a4_min: int = 420
    a4_max: int = 460
    identification_skip_stems: tuple[str, ...] = ("drums", "vocals")
    drum_stems: tuple[str, ...] = ("drums",)
    audio_sample_rate: int = 16000
    device: str = ""
