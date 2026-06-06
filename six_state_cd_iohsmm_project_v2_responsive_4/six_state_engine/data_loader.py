from __future__ import annotations
from pathlib import Path
from typing import Any
import zipfile
import pandas as pd
from .utils import normalize_columns, setup_logger

class InputZipLoader:
    def __init__(self, input_zip: str | Path, work_dir: str | Path, feature_contract: dict[str, Any]):
        self.input_zip = Path(input_zip)
        self.work_dir = Path(work_dir)
        self.feature_contract = feature_contract
        self.logger = setup_logger()
        self.extract_dir = self.work_dir / "_extracted_input"
        self.extract_dir.mkdir(parents=True, exist_ok=True)

    def extract(self) -> Path:
        if not self.input_zip.exists():
            raise FileNotFoundError(f"Input zip not found: {self.input_zip}")
        with zipfile.ZipFile(self.input_zip) as z:
            z.extractall(self.extract_dir)
        self.logger.info("Extracted input zip to %s", self.extract_dir)
        return self.extract_dir

    def find_csv_files(self) -> list[Path]:
        self.extract()
        return sorted(self.extract_dir.rglob("*.csv"))

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
