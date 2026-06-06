from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
import pandas as pd
from .utils import write_json

class FeatureContractError(RuntimeError):
    pass

@dataclass
class NoLeakageReport:
    passed: bool
    forbidden_columns_found: list[str]
    available_time_violations: dict[str, int]
    missing_required_columns: dict[str, list[str]]
    duplicate_time_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "forbidden_columns_found": self.forbidden_columns_found,
            "available_time_violations": self.available_time_violations,
            "missing_required_columns": self.missing_required_columns,
            "duplicate_time_count": self.duplicate_time_count,
            "notes": self.notes,
        }

def find_forbidden_columns(columns: Iterable[str], patterns: Iterable[str]) -> list[str]:
    out = []
    compiled = [re.compile(p) for p in patterns]
    for col in columns:
        if any(p.search(str(col)) for p in compiled):
            out.append(str(col))
    return sorted(set(out))

def validate_required_columns(df: pd.DataFrame, role: str, required: list[str]) -> list[str]:
    cols = set(df.columns)
    return [c for c in required if c not in cols]

def validate_available_lte_time(df: pd.DataFrame, available_col: str, time_col: str = "time") -> int:
    if available_col not in df.columns or time_col not in df.columns:
        return 0
    ok_mask = df[available_col].isna() | df[time_col].isna() | (df[available_col] <= df[time_col])
    return int((~ok_mask).sum())

def write_no_leakage_report(path: str | Path, report: NoLeakageReport) -> None:
    write_json(path, report.to_dict())
