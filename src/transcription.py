from pathlib import Path
from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH
from src.postprocessing import MidiPostProcessor

class AudioTranscriber:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.post_processor = MidiPostProcessor()

    def transcribe(self, stems, bpm):
        #Converts stems to MIDI and cleans/quantizes them.
        print("Transcribing stems to MIDI...")
        midi_paths = {}
        
        for stem_name, stem_path in stems.items():
            output_midi_path = self.output_dir / f"{stem_name}.mid"
            try:
                predict_and_save(
                    [stem_path],
                    str(self.output_dir),
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=False,
                    model_or_model_path=ICASSP_2022_MODEL_PATH
                )
                
                # Basic Pitch output filename logic: <input_stem>_basic_pitch.mid
                expected_output = self.output_dir / f"{Path(stem_path).stem}_basic_pitch.mid"
                final_midi_path = self.output_dir / f"{stem_name}.mid"
                
                if expected_output.exists():
                    expected_output.replace(final_midi_path)
                    
                    # Post-processing
                    self.post_processor.clean_and_quantize(str(final_midi_path), bpm)
                    
                    midi_paths[stem_name] = str(final_midi_path)
                else:
                    midi_paths[stem_name] = str(expected_output) 
                    
            except Exception as e:
                print(f"Error transcribing {stem_name}: {e}")
                import traceback
                traceback.print_exc()
                
        return midi_paths
