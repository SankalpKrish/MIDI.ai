from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from src.separation import StemSeparator
from src.analysis import GlobalAnalyzer
from src.transcription import AudioTranscriber
from src.identification import InstrumentClassifier
from src.postprocessing import MidiPostProcessor
from src.exceptions import (
    SeparationError,
    TranscriptionError,
    IdentificationError,
    AnalysisError,
    StageResult,
)

logger = logging.getLogger(__name__)

class AudioPipeline:
    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stems_dir = self.output_dir / "stems"
        self.midi_dir = self.output_dir / "midi"
        self.stems_dir.mkdir(exist_ok=True)
        self.midi_dir.mkdir(exist_ok=True)

        self.separator = StemSeparator(self.stems_dir)
        self.analyzer = GlobalAnalyzer()
        self.transcriber = AudioTranscriber(self.midi_dir)
        self.classifier = InstrumentClassifier()

    def process(self, input_file: str | Path) -> dict[str, Any]:
        results: dict[str, Any] = {}
        errors: list[dict[str, str]] = []

        if not os.path.exists(str(input_file)):
            raise FileNotFoundError(f"Input file {input_file} not found.")

        try:
            stems_dict = self.separator.separate(str(input_file))
            results['Stems'] = stems_dict
        except Exception as e:
            logger.error("Separation failed: %s", e, exc_info=True)
            raise SeparationError(str(e)) from e

        try:
            bpm_source = stems_dict.get('drums', str(input_file))
            results['Global Metadata'] = self.analyzer.analyze(
                key_source_path=str(input_file),
                bpm_source_path=bpm_source
            )
        except Exception as e:
            logger.error("Analysis failed: %s", e, exc_info=True)
            errors.append({'stage': 'analysis', 'error': str(e)})
            results['Global Metadata'] = {'bpm': 120.0, 'key': 'C Major', 'tuning_hz': 440.0}

        STEM_DEFAULTS: dict[str, dict[str, Any]] = {
            'bass': {'program': 34, 'program_name': 'Electric Bass (pick)', 'confidence': 1.0, 'matched_label': 'Bass stem'},
            'vocals': {'program': 52, 'program_name': 'Choir Aahs', 'confidence': 1.0, 'matched_label': 'Vocals stem'},
            'drums': {'program': 0, 'program_name': 'Percussion', 'confidence': 1.0, 'matched_label': 'Drums stem'},
        }

        identification_results: dict[str, Any] = {}
        try:
            logger.info("Assigning GM programs...")
            for stem_name, stem_path in stems_dict.items():
                if stem_name in STEM_DEFAULTS and stem_name != 'other':
                    identification_results[stem_name] = STEM_DEFAULTS[stem_name].copy()
                    r = identification_results[stem_name]
                    logger.info("  %s: Program %s (%s) [stem default]", stem_name, r['program'], r['program_name'])
                    continue
                try:
                    identification_results[stem_name] = self.classifier.get_midi_program(stem_path)
                    r = identification_results[stem_name]
                    logger.info("  %s: Program %s (%s)", stem_name, r['program'], r['program_name'])
                except Exception as e:
                    logger.warning("Identification failed for %s: %s", stem_name, e)
                    errors.append({'stage': 'identification', 'stem': stem_name, 'error': str(e)})
                    identification_results[stem_name] = {
                        'program': 0, 'program_name': 'Acoustic Grand Piano',
                        'confidence': 0.0, 'matched_label': 'Default'
                    }
        except Exception as e:
            logger.error("Identification stage error: %s", e, exc_info=True)
        results['Instrument Identification'] = identification_results

        bpm = float(results['Global Metadata']['bpm'])
        try:
            midi_paths, program_assignments = self.transcriber.transcribe(
                stems_dict, bpm, identification_results, drum_stems={'drums'}
            )
            results['Midi_Files'] = midi_paths
            results['GM_Program_Assignments'] = program_assignments
        except Exception as e:
            logger.error("Transcription failed: %s", e, exc_info=True)
            errors.append({'stage': 'transcription', 'error': str(e)})
            midi_paths = {}

        if midi_paths:
            try:
                merged_name = Path(str(input_file)).stem
                merged_path = str(self.output_dir / f"{merged_name}.mid")
                self.post_processor = MidiPostProcessor()
                self.post_processor.merge(
                    midi_paths, bpm, merged_path, is_drum_stems={'drums'}
                )
                results['Merged_Midi'] = merged_path
            except Exception as e:
                logger.warning("Merge failed: %s", e)
                errors.append({'stage': 'merge', 'error': str(e)})

        if errors:
            results['errors'] = errors
        return results
