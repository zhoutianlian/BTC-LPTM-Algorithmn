from __future__ import annotations
from pathlib import Path
from typing import Any
import shutil
import zipfile
import pandas as pd
from .utils import normalize_columns, setup_logger

class InputBundleLoader:
    def __init__(self, input_path: str | Path, work_dir: str | Path, feature_contract: dict[str, Any]):
        self.input_path = Path(input_path)
        self.work_dir = Path(work_dir)
        self.feature_contract = feature_contract
        self.logger = setup_logger()
        self.extract_dir = self.work_dir / "_extracted_input"

    def prepare_input_dir(self) -> Path:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        if self.input_path.is_dir():
            self.logger.info("Using input directory: %s", self.input_path)
            return self.input_path
        if not self.input_path.is_file() or not zipfile.is_zipfile(self.input_path):
            raise ValueError(f"Input path must be a zip file or directory: {self.input_path}")

        if self.extract_dir.exists():
            shutil.rmtree(self.extract_dir)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.input_path) as z:
            z.extractall(self.extract_dir)
        self.logger.info("Extracted input zip to %s", self.extract_dir)
        return self.extract_dir

    def find_csv_files(self) -> list[Path]:
        input_dir = self.prepare_input_dir()
        return sorted(input_dir.rglob("*.csv"))

    def match_roles(self) -> dict[str, Path | None]:
        csvs = self.find_csv_files()
        by_name = {p.name.lower(): p for p in csvs}
        roles: dict[str, Path | None] = {}
        for role, spec in self.feature_contract.get("files", {}).items():
            patterns = [p.lower() for p in spec.get("patterns", [])]
            required = bool(spec.get("required", False))
            selected = None
            for pat in patterns:
                if pat in by_name:
                    selected = by_name[pat]
                    break
            if selected is None:
                candidates = []
                for pat in patterns:
                    stem = pat.replace(".csv", "")
                    candidates.extend([p for p in csvs if stem in p.name.lower()])
                candidates = sorted(candidates, key=lambda p: ("copy" in p.name.lower(), len(p.name)))
                if candidates:
                    selected = candidates[0]
            if selected is None and required:
                raise FileNotFoundError(f"Required CSV role '{role}' not found. Expected patterns={patterns}")
            roles[role] = selected
            if selected:
                self.logger.info("Matched role %-20s -> %s", role, selected.name)
            else:
                self.logger.warning("Optional role %-20s missing", role)
        return roles

def read_csv_normalized(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    return normalize_columns(df)


# Backward-compatible name for older imports.
InputZipLoader = InputBundleLoader
