from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio
import src.utils as utils

logger = logging.getLogger(__name__)

class StemSeparator:
    def __init__(self, output_dir: str | Path, use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        utils.patch_torchaudio()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cache = use_cache
        logger.info("Loading Demucs (htdemucs) on %s...", self.device)
        self.model = get_model("htdemucs")
        self.model.to(self.device)
        self.model.eval()

    def separate(self, audio_path: str | Path) -> dict[str, str]:
        audio_path = Path(audio_path)
        logger.info("Separating sources for %s using Demucs...", audio_path)

        if self.use_cache:
            cached = self._check_cache(audio_path)
            if cached:
                return cached

        wav, sr_native = torchaudio.load(str(audio_path))

        ref = wav.mean(0)
        wav -= ref.mean()
        ref_std = ref.std()
        if ref_std > 0:
            wav /= ref_std

        wav = convert_audio(wav, sr_native, self.model.samplerate, self.model.audio_channels)

        logger.info("Running Demucs inference...")
        sources = apply_model(
            self.model,
            wav[None],
            device=self.device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True
        )[0]

        ref_std = ref.std()
        if ref_std > 0:
            sources *= ref_std
        sources += ref.mean()

        track_name = audio_path.stem
        output_folder = self.output_dir / "htdemucs" / track_name
        output_folder.mkdir(parents=True, exist_ok=True)

        stems_paths: dict[str, str] = {}
        model_sources = ["drums", "bass", "other", "vocals"]
        if hasattr(self.model, 'sources'):
            model_sources = self.model.sources

        for source, name in zip(sources, model_sources):
            stem_path = output_folder / f"{name}.wav"
            torchaudio.save(str(stem_path), source, self.model.samplerate)
            stems_paths[name] = str(stem_path)

        if self.use_cache:
            self._save_cache(audio_path, stems_paths)

        return stems_paths

    def _content_hash(self, audio_path: Path) -> str:
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _check_cache(self, audio_path: Path) -> dict[str, str] | None:
        manifest_path = self.output_dir / "stems.manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            file_hash = self._content_hash(audio_path)
            entry = manifest.get(audio_path.name)
            if entry and entry.get("hash") == file_hash:
                stem_paths = {}
                all_exist = True
                for name, path_str in entry.get("stems", {}).items():
                    p = Path(path_str)
                    if p.exists():
                        stem_paths[name] = str(p)
                    else:
                        all_exist = False
                        break
                if all_exist and stem_paths:
                    logger.info("Cache hit for %s", audio_path.name)
                    return stem_paths
        except Exception:
            pass
        return None

    def _save_cache(self, audio_path: Path, stems_paths: dict[str, str]) -> None:
        manifest_path = self.output_dir / "stems.manifest.json"
        manifest: dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        manifest[audio_path.name] = {
            "hash": self._content_hash(audio_path),
            "stems": stems_paths,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
