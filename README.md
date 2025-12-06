# MIDI.ai

MIDI.ai is a comprehensive audio analysis and transcription pipeline designed to convert raw audio files into structured MIDI data. It leverages state-of-the-art machine learning models for source separation, instrument identification, and polyphonic pitch detection, providing a robust tool for musicians, producers, and developers.

## Features

The pipeline performs the following sequential operations:

1.  **Source Separation**: Utilizes Demucs (Hybrid Transformer) to decompose the input audio into four distinct stems: Drums, Bass, Vocals, and Other.
2.  **Global Analysis**:
    - **BPM Detection**: Extracts tempo information, prioritizing the rhythmic clarity of the drums stem for high accuracy.
    - **Key & Tuning Estimation**: Detects the musical key (Major/Minor) and the tuning standard (reference frequency of A4) using spectral analysis.
3.  **Instrument Identification**: Analyzes the "Other" stem using YAMNet to identify dominant instruments and timbres.
4.  **Audio-to-MIDI Transcription**: Converts each separated stem into MIDI notes using Spotify's Basic Pitch model.
5.  **Post-Processing**:
    - **Quantization**: Aligns note start and end times to a strict 1/16th note grid based on the detected BPM.
    - **Cleaning**: Filters out ghost notes (duration < 0.1s) and low-confidence artifacts.
    - **Metadata**: Embeds correct tempo information directly into the MIDI file headers for seamless DAW integration.

## Installation

1.  Clone the repository.
2.  Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

**Note**: This project relies on several deep learning frameworks (TensorFlow, PyTorch) and audio processing libraries. A system with a dedicated GPU is recommended for optimal inference speed.

## Usage

Run the pipeline by providing the path to your input audio file (MP3 or WAV).

```bash
python audio_pipeline.py <path_to_audio_file>
```

### Optional Arguments

- `--output_dir`: Specify a custom directory for the output artifacts. Defaults to `output`.

```bash
python audio_pipeline.py mysong.mp3 --output_dir my_projects
```

## Output Structure

For an input file named `track.mp3`, the output directory will be structured as follows:

```
output/
  track/
    stems/
      htdemucs/
        track/
          drums.wav
          bass.wav
          vocals.wav
          other.wav
    midi/
      drums.mid
      bass.mid
      vocals.mid
      other.mid
```

The pipeline also prints a JSON summary of the analysis to the console, including detected BPM, Key, and instrument tags.

## Architecture

The project is structured as a modular Python package:

- `src/pipeline.py`: Main orchestration logic.
- `src/separation.py`: Demucs integration for source separation.
- `src/analysis.py`: Librosa-based global feature extraction.
- `src/transcription.py`: Basic Pitch integration for MIDI generation.
- `src/postprocessing.py`: MIDI data cleaning and quantization.
- `src/identification.py`: YAMNet integration for instrument classification.
