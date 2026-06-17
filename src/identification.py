from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
from typing import Any
import requests
import numpy as np

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow_hub as hub
import librosa

logger = logging.getLogger(__name__)

ProgramResult = dict[str, Any]

class InstrumentClassifier:
    def __init__(self):
        logger.info("Loading YAMNet model...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.yamnet_classes = self._load_yamnet_classes()
        self.gm_program_map = self._build_gm_program_map()

    def _load_yamnet_classes(self) -> np.ndarray:
        local_path = Path(__file__).parent / "data" / "yamnet_class_map.csv"
        class_names: list[str] = []

        if local_path.exists():
            with open(local_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    class_names.append(row[2])
            return np.array(class_names)

        url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            class_map_csv_text = response.text
            reader = csv.reader(io.StringIO(class_map_csv_text))
            next(reader)
            for row in reader:
                class_names.append(row[2])
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(class_map_csv_text, encoding="utf-8")
            return np.array(class_names)
        except Exception as e:
            raise RuntimeError(f"Failed to load YAMNet class map: {e}") from e

    def _build_gm_program_map(self) -> dict[str, int]:
        return {
            'Piano': 0, 'Electric piano': 4, 'Harpsichord': 6,
            'Marimba': 12, 'Xylophone': 13, 'Vibraphone': 14, 'Glockenspiel': 9,
            'Organ': 16, 'Hammond organ': 16, 'Church organ': 19, 'Accordion': 21, 'Harmonica': 22,
            'Guitar': 24, 'Electric guitar': 27, 'Acoustic guitar': 24, 'Bass guitar': 33,
            'Steel guitar': 28, 'Ukulele': 23,
            'Bass': 33, 'Electric bass': 34, 'Acoustic bass': 32, 'Double bass': 32,
            'Violin': 40, 'Cello': 42, 'String section': 48,
            'Trumpet': 56, 'Trombone': 57, 'French horn': 60, 'Brass instrument': 60,
            'Saxophone': 65, 'Clarinet': 71,
            'Flute': 73,
            'Synthesizer': 80,
            'Drum': 0, 'Drum kit': 0, 'Cymbal': 0,
        }

    def _predict(self, audio_path: str | Path) -> np.ndarray:
        wav_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        peak = np.max(np.abs(wav_data))
        if peak > 0:
            wav_data = wav_data / peak
        scores, embeddings, spectrogram = self.yamnet_model(wav_data)
        prediction = np.mean(scores, axis=0)
        return prediction

    def get_midi_program(self, audio_path: str | Path) -> ProgramResult:
        logger.info("Getting MIDI program for %s...", audio_path)
        prediction = self._predict(audio_path)

        top_n = 10
        top_indices = np.argsort(prediction)[::-1][:top_n]
        top_labels = self.yamnet_classes[top_indices]
        top_scores = prediction[top_indices]

        best: tuple[float, int, str, str] = (0.0, 0, '', '')
        for label, score in zip(top_labels, top_scores):
            label_lower = label.lower()
            for key, program in self.gm_program_map.items():
                if key.lower() in label_lower:
                    match_len = len(key)
                    if match_len > best[1] or (match_len == best[1] and score > best[0]):
                        best = (float(score), match_len, label, key)

        if best[2]:
            return {
                'program': self.gm_program_map[best[3]],
                'program_name': self._get_gm_program_name(self.gm_program_map[best[3]]),
                'confidence': best[0],
                'matched_label': best[2],
            }

        return {
            'program': 0,
            'program_name': 'Acoustic Grand Piano',
            'confidence': 0.0,
            'matched_label': 'Default'
        }

    def _get_gm_program_name(self, program_number: int) -> str:
        program_names = {
            0: 'Acoustic Grand Piano', 1: 'Bright Acoustic Piano', 2: 'Electric Grand Piano',
            3: 'Honky-tonk Piano', 4: 'Electric Piano 1', 5: 'Electric Piano 2',
            6: 'Harpsichord', 7: 'Clavi', 8: 'Celesta', 9: 'Glockenspiel',
            10: 'Music Box', 11: 'Vibraphone', 12: 'Marimba', 13: 'Xylophone',
            14: 'Tubular Bells', 15: 'Dulcimer', 16: 'Drawbar Organ', 17: 'Percussive Organ',
            18: 'Rock Organ', 19: 'Church Organ', 20: 'Reed Organ', 21: 'Accordion',
            22: 'Harmonica', 23: 'Tango Accordion', 24: 'Acoustic Guitar (nylon)',
            25: 'Acoustic Guitar (steel)', 26: 'Electric Guitar (jazz)', 27: 'Electric Guitar (clean)',
            28: 'Electric Guitar (muted)', 29: 'Overdriven Guitar', 30: 'Distortion Guitar',
            31: 'Guitar Harmonics', 32: 'Acoustic Bass', 33: 'Electric Bass (finger)',
            34: 'Electric Bass (pick)', 35: 'Fretless Bass', 36: 'Slap Bass 1',
            37: 'Slap Bass 2', 38: 'Synth Bass 1', 39: 'Synth Bass 2',
            40: 'Violin', 41: 'Viola', 42: 'Cello', 43: 'Contrabass',
            44: 'Tremolo Strings', 45: 'Pizzicato Strings', 46: 'Orchestral Harp', 47: 'Timpani',
            48: 'String Ensemble 1', 49: 'String Ensemble 2', 50: 'Synth Strings 1',
            51: 'Synth Strings 2', 52: 'Choir Aahs', 53: 'Voice Oohs', 54: 'Synth Choir',
            55: 'Orchestra Hit', 56: 'Trumpet', 57: 'Trombone', 58: 'Tuba',
            59: 'Muted Trumpet', 60: 'French Horn', 61: 'Brass Section', 62: 'Synth Brass 1',
            63: 'Synth Brass 2', 64: 'Soprano Sax', 65: 'Alto Sax', 66: 'Tenor Sax',
            67: 'Baritone Sax', 68: 'Oboe', 69: 'English Horn', 70: 'Bassoon',
            71: 'Clarinet', 72: 'Piccolo', 73: 'Flute', 74: 'Recorder',
            75: 'Pan Flute', 76: 'Blown Bottle', 77: 'Shakuhachi', 78: 'Whistle',
            79: 'Ocarina', 80: 'Lead 1 (square)', 81: 'Lead 2 (sawtooth)', 82: 'Lead 3 (calliope)',
            83: 'Lead 4 (chiff)', 84: 'Lead 5 (charang)', 85: 'Lead 6 (voice)',
            86: 'Lead 7 (fifths)', 87: 'Lead 8 (bass + lead)', 88: 'Pad 1 (new age)',
            89: 'Pad 2 (warm)', 90: 'Pad 3 (polysynth)', 91: 'Pad 4 (choir)',
            92: 'Pad 5 (bowed)', 93: 'Pad 6 (metallic)', 94: 'Pad 7 (halo)',
            95: 'Pad 8 (sweep)', 96: 'FX 1 (rain)', 97: 'FX 2 (soundtrack)',
            98: 'FX 3 (crystal)', 99: 'FX 4 (atmosphere)', 100: 'FX 5 (brightness)',
            101: 'FX 6 (goblins)', 102: 'FX 7 (echoes)', 103: 'FX 8 (sci-fi)',
            104: 'Sitar', 105: 'Banjo', 106: 'Shamisen', 107: 'Koto',
            108: 'Kalimba', 109: 'Bagpipe', 110: 'Fiddle', 111: 'Shanai',
            112: 'Tinkle Bell', 113: 'Agogo', 114: 'Steel Drums', 115: 'Woodblock',
            116: 'Taiko Drum', 117: 'Melodic Tom', 118: 'Synth Drum', 119: 'Reverse Cymbal',
            120: 'Guitar Fret Noise', 121: 'Breath Noise', 122: 'Seashore', 123: 'Bird Tweet',
            124: 'Telephone Ring', 125: 'Helicopter', 126: 'Applause', 127: 'Gunshot'
        }
        return program_names.get(program_number, 'Unknown')
