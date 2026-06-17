import pretty_midi
import numpy as np
import tempfile
from pathlib import Path
from src.postprocessing import MidiPostProcessor


def _make_midi(path, notes=None, bpm=120, is_drum=False):
    if notes is None:
        notes = [(60, 0.0, 1.0)]
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=is_drum)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


class TestCleanAndQuantize:
    def setup_method(self):
        self.pp = MidiPostProcessor()
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_ghost_notes_removed_after_quantize(self):
        midi_path = _make_midi(
            self.tmpdir / "ghost.mid",
            notes=[(60, 0.001, 0.0029)],
            bpm=120,
        )
        self.pp.clean_and_quantize(midi_path, bpm=120)
        pm = pretty_midi.PrettyMIDI(midi_path)
        total = sum(len(i.notes) for i in pm.instruments)
        assert total == 0, "zero-length notes after quantize should be removed"

    def test_normal_note_survives(self):
        midi_path = _make_midi(
            self.tmpdir / "normal.mid",
            notes=[(60, 0.0, 1.0)],
            bpm=120,
        )
        self.pp.clean_and_quantize(midi_path, bpm=120)
        pm = pretty_midi.PrettyMIDI(midi_path)
        total = sum(len(i.notes) for i in pm.instruments)
        assert total == 1

    def test_quantize_snaps_to_grid(self):
        midi_path = _make_midi(
            self.tmpdir / "snap.mid",
            notes=[(60, 0.03, 0.27)],
            bpm=120,
        )
        self.pp.clean_and_quantize(midi_path, bpm=120)
        pm = pretty_midi.PrettyMIDI(midi_path)
        note = pm.instruments[0].notes[0]
        sixteenth = 60.0 / 120 / 4
        assert abs(note.start - round(note.start / sixteenth) * sixteenth) < 1e-9
        assert abs(note.end - round(note.end / sixteenth) * sixteenth) < 1e-9

    def test_program_assignment(self):
        midi_path = _make_midi(self.tmpdir / "prog.mid", bpm=120)
        self.pp.clean_and_quantize(midi_path, bpm=120, program_number=33)
        pm = pretty_midi.PrettyMIDI(midi_path)
        assert pm.instruments[0].program == 33

    def test_is_drum_flag(self):
        midi_path = _make_midi(self.tmpdir / "drum.mid", bpm=120)
        self.pp.clean_and_quantize(midi_path, bpm=120, is_drum=True)
        pm = pretty_midi.PrettyMIDI(midi_path)
        assert pm.instruments[0].is_drum


class TestMerge:
    def setup_method(self):
        self.pp = MidiPostProcessor()
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_merge_two_stems(self):
        drum_path = _make_midi(self.tmpdir / "drums.mid", notes=[(36, 0.0, 0.5)], is_drum=True)
        bass_path = _make_midi(self.tmpdir / "bass.mid", notes=[(40, 0.0, 1.0)])
        out = str(self.tmpdir / "merged.mid")
        self.pp.merge({"drums": drum_path, "bass": bass_path}, bpm=120, out_path=out)
        pm = pretty_midi.PrettyMIDI(out)
        assert len(pm.instruments) == 2

    def test_merge_drums_set_is_drum(self):
        drum_path = _make_midi(self.tmpdir / "d2.mid", notes=[(36, 0.0, 0.5)])
        out = str(self.tmpdir / "merged2.mid")
        self.pp.merge({"drums": drum_path}, bpm=120, out_path=out, is_drum_stems={"drums"})
        pm = pretty_midi.PrettyMIDI(out)
        assert pm.instruments[0].is_drum

    def test_merge_preserves_notes(self):
        a_path = _make_midi(self.tmpdir / "a.mid", notes=[(60, 0.0, 1.0)])
        b_path = _make_midi(self.tmpdir / "b.mid", notes=[(64, 0.5, 1.5)])
        out = str(self.tmpdir / "merged3.mid")
        self.pp.merge({"a": a_path, "b": b_path}, bpm=120, out_path=out)
        pm = pretty_midi.PrettyMIDI(out)
        all_notes = [n for i in pm.instruments for n in i.notes]
        assert len(all_notes) == 2
