from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from .config import load_run_config
from .pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG = PROJECT_ROOT / "configs" / "run_config.yaml"


def _resolve_cli_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _path_base(run_cfg: dict[str, Any], run_config_path: Path) -> Path:
    base_name = str(run_cfg.get("path_base", "project_root")).strip().lower()
    if base_name == "project_root":
        return PROJECT_ROOT
    if base_name == "config_dir":
        return run_config_path.parent
    if base_name == "cwd":
        return Path.cwd()
    raise ValueError(
        "Unsupported run.path_base value "
        f"{base_name!r}. Expected one of: project_root, config_dir, cwd."
    )


def _resolve_config_path(value: Any, base_dir: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Six-State CD-IOHSMM market state engine.")
    parser.add_argument(
        "--run-config",
        default=None,
        help=f"Run config YAML. Defaults to {DEFAULT_RUN_CONFIG}",
    )
    parser.add_argument(
        "--input-path",
        "--input-zip",
        dest="input_path",
        default=None,
        help="Path to an input .zip file or a directory containing CSV files",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for outputs")
    parser.add_argument("--config-dir", default=None, help="Config directory. Defaults to project configs/")
    args = parser.parse_args()

    run_config_path = _resolve_cli_path(args.run_config) if args.run_config else DEFAULT_RUN_CONFIG
    run_cfg_raw = load_run_config(run_config_path) if run_config_path.exists() else {}
    run_cfg = run_cfg_raw.get("run", run_cfg_raw)
    if not isinstance(run_cfg, dict):
        raise ValueError(f"Run config must contain a mapping: {run_config_path}")

    config_base = _path_base(run_cfg, run_config_path)
    input_path = _resolve_cli_path(args.input_path) or _resolve_config_path(run_cfg.get("input_path"), config_base)
    output_dir = _resolve_cli_path(args.output_dir) or _resolve_config_path(run_cfg.get("output_dir"), config_base)
    config_dir = _resolve_cli_path(args.config_dir) or _resolve_config_path(run_cfg.get("config_dir"), config_base)

    if input_path is None:
        parser.error("Missing input path. Set run.input_path in configs/run_config.yaml or pass --input-path.")
    if output_dir is None:
        parser.error("Missing output dir. Set run.output_dir in configs/run_config.yaml or pass --output-dir.")
    if config_dir is None:
        config_dir = PROJECT_ROOT / "configs"

    result = run_pipeline(
        input_zip=input_path,
        output_dir=output_dir,
        config_dir=config_dir,
    )
    print("Pipeline completed.")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
