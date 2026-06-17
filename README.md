# MIDI.ai

MIDI.ai is a modular audio processing framework that converts raw polyphonic audio into structured MIDI data, using deep learning models for source separation, transcription, and instrument classification.

## Project Overview

The system breaks Automatic Music Transcription (AMT) into specialized sub-tasks:

1. Isolate individual instruments via source separation
2. Analyze global musical features (tempo, key, tuning)
3. Classify instrument timbres and assign General MIDI programs
4. Transcribe each stem to MIDI with quantization and artifact removal
5. Merge all stems into a single synchronized multi-track MIDI file

## Pipeline Stages

### 1. Source Separation (Demucs)

Input audio is separated into **Drums, Bass, Vocals, Other** using Hybrid Transformer Demucs (v4). The model is loaded once per pipeline instance (not per file), and separated stems are cached via SHA256 content hash so re-processing skips re-separation.

### 2. Music Information Retrieval (Librosa)

- **BPM**: Detected from the Drums stem for clearest transient information
- **Key**: Krumhansl-Schmuckler profile correlation on chroma features
- **Tuning**: A4 reference estimation (defaults to 440Hz)

### 3. Instrument Classification

- **Bass stem** → Electric Bass (pick) — GM 34
- **Vocals stem** → Choir Aahs — GM 52
- **Other stem** → Classified by YAMNet (TF-Hub) with longest-match GM mapping
- **Drums** → Routed to MIDI percussion channel (channel 10)

YAMNet class map is bundled in-repo (`src/data/yamnet_class_map.csv`); no runtime network fetch.

### 4. Transcription (Basic Pitch)

All stems are transcribed in a single batched `predict_and_save` call. Basic Pitch is lazy-imported at call time to avoid forcing TF graph construction at module load.

### 5. Post-Processing

- **Quantization**: Notes snapped to a 1/16th note grid (after quantization, zero-length notes removed)
- **Ghost-note filter**: Applied post-quantization to avoid creating zero-duration artifacts
- **Program Assignment**: GM programs assigned per stem based on classification

### 6. Merge

All per-stem MIDI files are merged into a single multi-track `.mid` file with correct BPM, program numbers, and drum channel assignment.

## Output

```
output/<input_name>/
├── stems/
│   └── htdemucs/<input_name>/
│       ├── drums.wav
│       ├── bass.wav
│       ├── other.wav
│       └── vocals.wav
├── midi/
│   ├── drums.mid
│   ├── bass.mid
│   ├── other.mid
│   └── vocals.mid
└── <input_name>.mid          # merged multi-track output
```

## Architecture

- `audio_pipeline.py` — CLI entry point (single file, directory, or glob)
- `src/pipeline.py` — Orchestrator; error-handled per-stage execution
- `src/separation.py` — Demucs inference + stem caching
- `src/analysis.py` — BPM, key, tuning estimation
- `src/identification.py` — YAMNet instrument classifier + GM mapping
- `src/transcription.py` — Basic Pitch transcription + post-processing
- `src/postprocessing.py` — Quantization, ghost-note removal, merge
- `src/config.py` — Centralized `Config` dataclass
- `src/exceptions.py` — Custom exception hierarchy + `StageResult`
- `src/logging_config.py` — Logging setup (stderr); JSON output on stdout
- `src/utils.py` — NumpyEncoder for JSON serialization, torchaudio monkeypatch
- `tests/` — pytest suite (post-processing, merge, utilities)

## Installation

```bash
git clone https://github.com/SankalpKrish/MIDI.ai.git
cd MIDI.ai
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

A GPU with CUDA support is recommended for TensorFlow and PyTorch components.

## Usage

```bash
python audio_pipeline.py <input> [--output_dir OUTPUT] [--verbose] [--no-cache]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Path to audio file, directory of audio files, or glob pattern |
| `--output_dir` | Output directory (default: `output/`) |
| `--verbose`, `-v` | Enable debug logging |
| `--no-cache` | Skip stem cache (re-run Demucs) |

**Examples:**

```bash
# Single file
python audio_pipeline.py song.wav

# Directory (processes all .wav/.mp3/.flac)
python audio_pipeline.py input/ --verbose

# Glob pattern
python audio_pipeline.py "tracks/*.mp3"

# Batch results written to output/_batch_results.json
```

**As an installed CLI:**

```bash
midi-ai song.wav --verbose
```

## Docker

```bash
docker build -t midi-ai .
docker run --gpus all -v $(pwd)/input:/input -v $(pwd)/output:/output midi-ai /input/song.wav --output_dir /output
```

## CI

GitHub Actions runs ruff linting, pytest, and mypy on push/PR (ubuntu + windows).

## Future Improvements

- VST plugin for DAW integration
- Swing/triplet quantization
- MusicXML export
- Chord detection / harmonic annotation
- Web API (FastAPI)
- Real-time / streaming transcription
