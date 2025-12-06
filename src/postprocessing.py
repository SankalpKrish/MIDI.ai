import pretty_midi

class MidiPostProcessor:
    def clean_and_quantize(self, midi_path, bpm):
        """
        Post-processes the MIDI file:
        1. Resets the MIDI object with correct BPM.
        2. Filter ghost notes (< 0.1s).
        3. Quantize to 1/16th grid.
        """
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            
            # Create fresh PrettyMIDI with correct BPM
            new_pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            
            sixteenth_duration = 60.0 / bpm / 4.0
            
            for instrument in pm.instruments:
                new_inst = pretty_midi.Instrument(
                    program=instrument.program, 
                    is_drum=instrument.is_drum, 
                    name=instrument.name
                )
                
                cleaned_notes = []
                for note in instrument.notes:
                    if (note.end - note.start) < 0.1:
                        continue
                        
                    # Quantization
                    note.start = round(note.start / sixteenth_duration) * sixteenth_duration
                    note.end = round(note.end / sixteenth_duration) * sixteenth_duration
                    
                    cleaned_notes.append(note)
                
                new_inst.notes = cleaned_notes
                new_inst.pitch_bends = instrument.pitch_bends
                new_inst.control_changes = instrument.control_changes
                new_pm.instruments.append(new_inst)
                
            new_pm.write(midi_path)
            print(f"Cleaned and Quantized {midi_path} (BPM: {bpm})")
            
        except Exception as e:
            print(f"Failed to clean MIDI {midi_path}: {e}")
