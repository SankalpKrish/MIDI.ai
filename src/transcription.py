from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from src.postprocessing import MidiPostProcessor

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.post_processor = MidiPostProcessor()

    def transcribe(self, stems: dict[str, str | Path], bpm: float, identification_results: dict[str, Any] | None = None, drum_stems: set[str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
        if drum_stems is None:
            drum_stems = set()
        from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH

        logger.info("Transcribing stems to MIDI...")
        midi_paths: dict[str, str] = {}
        program_assignments: dict[str, Any] = {}

        stem_paths = list(stems.values())
        try:
            predict_and_save(
                [str(p) for p in stem_paths],
                str(self.output_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH
            )
        except Exception as e:
            logger.error("Batch transcription failed, falling back to per-stem: %s", e)
            stem_paths = []

        for stem_name, stem_path in stems.items():
            expected_output = self.output_dir / f"{Path(str(stem_path)).stem}_basic_pitch.mid"
            final_midi_path = self.output_dir / f"{stem_name}.mid"

            if expected_output.exists():
                try:
                    expected_output.replace(final_midi_path)
                    program_number = None
                    if identification_results and stem_name in identification_results:
                        program_number = identification_results[stem_name].get('program')

                    is_drum = stem_name in drum_stems
                    self.post_processor.clean_and_quantize(
                        str(final_midi_path), bpm, program_number, is_drum=is_drum
                    )

                    midi_paths[stem_name] = str(final_midi_path)
                    if program_number is not None:
                        program_assignments[stem_name] = {
                            'program': program_number,
                            'program_name': identification_results[stem_name].get('program_name'),
                            'confidence': identification_results[stem_name].get('confidence'),
                            'matched_label': identification_results[stem_name].get('matched_label')
                        }
                except Exception as e:
                    logger.error("Error processing %s: %s", stem_name, e, exc_info=True)
            else:
                logger.warning("No output for %s, expected %s", stem_name, expected_output)

        return midi_paths, program_assignments
