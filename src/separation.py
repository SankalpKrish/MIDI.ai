from pathlib import Path
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio
import src.utils as utils

class StemSeparator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        utils.patch_torchaudio()

    def separate(self, audio_path):
        # Separate audio into 4 stems: drums, bass, vocals, other.
        print(f"Separating sources for {audio_path} using Demucs (In-Process)...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Demucs inference device: {device}")

        # Load Model (htdemucs is v4 default)
        model = get_model("htdemucs")
        model.to(device)
        model.eval()

        # Load Audio
        wav, sr_native = torchaudio.load(str(audio_path))
        
        # Normalize/Preprocess
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        
        wav = convert_audio(wav, sr_native, model.samplerate, model.audio_channels)
        
        # Inference
        print("Running inference...")
        sources = apply_model(
            model, 
            wav[None], 
            device=device, 
            shifts=1, 
            split=True, 
            overlap=0.25, 
            progress=True
        )[0]
        
        sources *= ref.std()
        sources += ref.mean()

        # Save Stems
        track_name = Path(audio_path).stem
        output_folder = self.output_dir / "htdemucs" / track_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        stems_paths = {}
        model_sources = ["drums", "bass", "other", "vocals"]
        if hasattr(model, 'sources'):
            model_sources = model.sources

        for source, name in zip(sources, model_sources):
            stem_path = output_folder / f"{name}.wav"
            torchaudio.save(str(stem_path), source, model.samplerate)
            stems_paths[name] = str(stem_path)
            
        return stems_paths
