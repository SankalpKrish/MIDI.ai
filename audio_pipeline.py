import argparse
import glob
import json
import logging
from pathlib import Path
from src.pipeline import AudioPipeline
from src.utils import NumpyEncoder
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

def _resolve_inputs(input_arg: str) -> list[str]:
    p = Path(input_arg)
    if p.is_dir():
        files: list[str] = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
            files.extend(str(f) for f in p.glob(ext))
            files.extend(str(f) for f in p.glob(ext.upper()))
        if not files:
            raise FileNotFoundError(f"No audio files found in directory: {input_arg}")
        files.sort()
        return files
    if p.is_file():
        return [str(p)]
    matched = glob.glob(input_arg)
    if matched:
        return sorted(matched)
    raise FileNotFoundError(f"Input not found: {input_arg}")

def main():
    parser = argparse.ArgumentParser(description="Audio Analysis Pipeline")
    parser.add_argument("input", help="Path to audio file, directory, or glob pattern")
    parser.add_argument("--output_dir", default="output", help="Directory to save output files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-cache", action="store_true", help="Skip stem cache")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    inputs = _resolve_inputs(args.input)
    all_results = {}

    for input_file in inputs:
        input_stem = Path(input_file).stem
        project_output_dir = Path(args.output_dir) / input_stem
        logger.info("Processing %s → %s", input_file, project_output_dir)

        pipeline = AudioPipeline(output_dir=project_output_dir)
        try:
            data = pipeline.process(input_file)
            all_results[input_file] = data
            print(json.dumps({input_file: data}, indent=4, cls=NumpyEncoder))
        except Exception as e:
            logger.exception("Pipeline failed for %s: %s", input_file, e)
            all_results[input_file] = {"error": str(e)}

    if len(inputs) > 1:
        summary_path = Path(args.output_dir) / "_batch_results.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(all_results, indent=4, cls=NumpyEncoder), encoding="utf-8"
        )
        logger.info("Batch results written to %s", summary_path)

if __name__ == "__main__":
    main()
