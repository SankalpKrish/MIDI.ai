# MIDI.ai

MIDI.ai is a modular audio processing framework developed to explore the intersection of Digital Signal Processing (DSP) and Machine Learning. The project implements a robust end-to-end pipeline for converting raw polyphonic audio into structured MIDI data, utilizing state-of-the-art deep learning models for source separation, transcription, and classification.

Designed as an advanced personal project, this repository demonstrates the integration of multiple complex neural networks into a cohesive, production-ready Python application.

## Project Overview

The primary objective of MIDI.ai is to solve the complex problem of Automatic Music Transcription (AMT) by breaking it down into specialized sub-tasks. Rather than relying on a single monolithic model, the system employs a "divide and conquer" strategy:

1.  Isolating individual instruments to reduce spectral overlapping.
2.  Analyzing global musical features (Tempo, Key) from the most reliable sources.
3.  Transcribing and recombining these streams into a synchronized MIDI file.

## Technical Methodology

The pipeline executes the following stages sequentially:

### 1. Source Separation (Demucs)

The input audio is first processed using the **Hybrid Transformer Demucs (v4)** model. This deep learning architecture allows for high-fidelity separation of the mix into four constituent stems: Drums, Bass, Vocals, and Other. This step is critical for reducing interference during pitch detection.

### 2. Music Information Retrieval (MIR)

Global metadata is extracted using **Librosa** and signal processing techniques:

- **BPM Detection**: To ensure rhythmic accuracy, tempo is detected specifically from the _Drums_ stem, which provides the clearest transient information.
- **Key & Tuning**: Spectral chromagrams are generated to estimate the musical key and the reference tuning frequency (e.g., A4 = 440Hz).

### 3. Classification (YAMNet)

The pipeline employs **YAMNet**, a pre-trained deep neural network, to classify the timbre of the "Other" stem. This adds semantic understanding to the transcription, identifying whether the accompaniment consists of guitars, pianos, or synthesizers.

### 4. Transcription (Basic Pitch)

Audio-to-MIDI conversion is handled by **Basic Pitch**, a lightweight yet powerful model optimized for instrument transcription. It predicts pitch events with high time-frequency resolution.

### 5. Algorithmic Post-Processing

Raw model outputs are rarely perfect. A custom post-processing layer applies musical heuristics:

- **Quantization**: Note onsets and offsets are snapped to a 1/16th note grid derived from the detected BPM.
- **Artifact Removal**: Short-duration "ghost notes" (likely false positives) are algorithmically filtered out to produce a cleaner score.

## Software Architecture

The codebase has been refactored from a monolithic script into a modular package structure to ensure maintainability and scalability.

- `src.pipeline`: Orchestrator that manages data flow between subsystems.
- `src.separation`: Encapsulates the Demucs inference logic.
- `src.analysis`: Handles MIR tasks and DSP algorithms.
- `src.transcription`: Manages the interface with the Basic Pitch library.
- `src.postprocessing`: Contains the custom quantization and cleaning logic.
- `src.identification`: Wraps the TensorFlow Hub YAMNet model.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MIDI.ai.git
    cd MIDI.ai
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

_Note: Execution requires significant computational resources. A GPU with CUDA support is highly recommended for the TensorFlow and PyTorch components._

## Usage

The pipeline can be executed via the command line interface:

```bash
python audio_pipeline.py <input_audio_file>
```

**Arguments:**

- `input_file`: Path to the .mp3 or .wav file to transcribe.
- `--output_dir`: (Optional) Directory for generated stems and MIDI files.

## Future Improvements

- Implement VST plugin support for direct DAW integration.
- Improve quantization logic to support swing and triplet grids.
- Fine-tune instrument recognition to automatically assign MIDI Program Change messages.
