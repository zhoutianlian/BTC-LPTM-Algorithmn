from __future__ import annotations
import argparse
from pathlib import Path
from .pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Six-State CD-IOHSMM market state engine.")
    parser.add_argument("--input-zip", required=True, help="Path to input.zip")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--config-dir", default=None, help="Config directory. Defaults to project configs/")
    args = parser.parse_args()
    result = run_pipeline(
        input_zip=Path(args.input_zip),
        output_dir=Path(args.output_dir),
        config_dir=Path(args.config_dir) if args.config_dir else None,
    )
    print("Pipeline completed.")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
