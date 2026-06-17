from __future__ import annotations

import logging
from pathlib import Path
import pretty_midi

logger = logging.getLogger(__name__)

class MidiPostProcessor:
    def assign_program_number(self, midi_path: str | Path, program_number: int) -> bool:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            for instrument in pm.instruments:
                if not instrument.is_drum:
                    instrument.program = program_number
            pm.write(str(midi_path))
            logger.info("Assigned program %d to %s", program_number, midi_path)
            return True
        except Exception as e:
            logger.error("Failed to assign program to %s: %s", midi_path, e)
            return False

    def clean_and_quantize(self, midi_path: str | Path, bpm: float, program_number: int | None = None, is_drum: bool | None = None) -> None:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            new_pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            sixteenth_duration = 60.0 / bpm / 4.0

            for instrument in pm.instruments:
                drum_flag = is_drum if is_drum is not None else instrument.is_drum
                target_program = program_number if (program_number is not None and not drum_flag) else instrument.program

                new_inst = pretty_midi.Instrument(
                    program=target_program,
                    is_drum=drum_flag,
                    name=instrument.name
                )

                cleaned_notes: list[pretty_midi.Note] = []
                for note in instrument.notes:
                    note.start = round(note.start / sixteenth_duration) * sixteenth_duration
                    note.end = round(note.end / sixteenth_duration) * sixteenth_duration
                    if note.end <= note.start:
                        continue
                    cleaned_notes.append(note)

                new_inst.notes = cleaned_notes
                new_inst.pitch_bends = instrument.pitch_bends
                new_inst.control_changes = instrument.control_changes
                new_pm.instruments.append(new_inst)

            new_pm.write(str(midi_path))
            logger.info("Cleaned and quantized %s (BPM: %s, Program: %s)", midi_path, bpm, target_program)

        except Exception as e:
            logger.error("Failed to clean MIDI %s: %s", midi_path, e)

    def merge(self, midi_paths: dict[str, str | Path], bpm: float, out_path: str | Path, is_drum_stems: set[str] | None = None) -> str:
        if is_drum_stems is None:
            is_drum_stems = set()
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        for stem_name, stem_path in midi_paths.items():
            try:
                stem_pm = pretty_midi.PrettyMIDI(str(stem_path))
                for inst in stem_pm.instruments:
                    merged_inst = pretty_midi.Instrument(
                        program=inst.program,
                        is_drum=(stem_name in is_drum_stems) or inst.is_drum,
                        name=stem_name,
                    )
                    merged_inst.notes = inst.notes
                    merged_inst.pitch_bends = inst.pitch_bends
                    merged_inst.control_changes = inst.control_changes
                    pm.instruments.append(merged_inst)
            except Exception as e:
                logger.warning("Skipping %s during merge: %s", stem_name, e)
        pm.write(str(out_path))
        logger.info("Merged %d stems into %s", len(midi_paths), out_path)
        return str(out_path)
