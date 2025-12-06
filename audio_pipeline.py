import argparse
import json
from pathlib import Path
from src.pipeline import AudioPipeline
from src.utils import NumpyEncoder

def main():
    parser = argparse.ArgumentParser(description="Audio Analysis Pipeline")
    parser.add_argument("input_file", help="Path to the input audio file (MP3/WAV)")
    parser.add_argument("--output_dir", default="output", help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Create a unique output directory for this input file to avoid conflicts
    input_stem = Path(args.input_file).stem
    project_output_dir = Path(args.output_dir) / input_stem
    
    pipeline = AudioPipeline(output_dir=project_output_dir)
    
    try:
        data = pipeline.process(args.input_file)
        print(json.dumps(data, indent=4, cls=NumpyEncoder))
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
