from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

def setup_logger(name: str = "six_state_engine", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def normalize_col(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col(c) for c in out.columns]
    return out

def ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True).dt.tz_localize(None)
    return out

def safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def softmax_matrix(x: np.ndarray, temperature: float = 1.0, floor: float = 1e-12) -> np.ndarray:
    z = x / max(temperature, 1e-9)
    z = z - np.nanmax(z, axis=1, keepdims=True)
    e = np.exp(z)
    e = np.where(np.isfinite(e), e, 0.0)
    e = np.maximum(e, floor)
    return e / e.sum(axis=1, keepdims=True)

def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def write_json(path: str | Path, data: dict) -> None:
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_safe)
