import os
from pathlib import Path
from src.separation import StemSeparator
from src.analysis import GlobalAnalyzer
from src.transcription import AudioTranscriber
from src.identification import InstrumentClassifier

class AudioPipeline:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-component directories
        self.stems_dir = self.output_dir / "stems"
        self.midi_dir = self.output_dir / "midi"
        self.stems_dir.mkdir(exist_ok=True)
        self.midi_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.separator = StemSeparator(self.stems_dir)
        self.analyzer = GlobalAnalyzer()
        self.transcriber = AudioTranscriber(self.midi_dir)
        self.classifier = InstrumentClassifier()

    def process(self, input_file):
        # Main pipeline execution.
        results = {}
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found.")

        # 1. Source Separation (Demucs)
        # Using the drums stem significantly improves BPM detection accuracy.
        stems_dict = self.separator.separate(input_file)
        results['Stems'] = stems_dict

        # 2. Global Analysis (Librosa)
        # Prefer drums for BPM if available, as they have clearer transients.
        bpm_source = stems_dict.get('drums', input_file)
        results['Global Metadata'] = self.analyzer.analyze(
            key_source_path=input_file, 
            bpm_source_path=bpm_source
        )
        
        # 3. Instrument Identification (YAMNet on 'Other' stem)
        if 'other' in stems_dict:
            results['Instrumentation'] = self.classifier.identify(stems_dict['other'])
        else:
            results['Instrumentation'] = []

        # 4. Transcription & Post-Processing (Basic Pitch + PrettyMIDI)
        bpm = results['Global Metadata']['bpm']
        results['Midi_Files'] = self.transcriber.transcribe(stems_dict, bpm)
        
        return results
