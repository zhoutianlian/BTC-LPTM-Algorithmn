from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
import pandas as pd

@dataclass
class QuantileScaler:
    q_low: float = 0.05
    q_high: float = 0.95
    params: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: Iterable[str], train_mask: pd.Series | None = None) -> "QuantileScaler":
        d = df.loc[train_mask] if train_mask is not None else df
        for col in columns:
            if col not in d.columns:
                continue
            s = pd.to_numeric(d[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                self.params[col] = (0.0, 1.0)
                continue
            lo = float(s.quantile(self.q_low))
            hi = float(s.quantile(self.q_high))
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            self.params[col] = (lo, hi)
        return self

    def transform01(self, df: pd.DataFrame, col: str, default: float = 0.0, abs_value: bool = False) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if abs_value:
            s = s.abs()
        lo, hi = self.params.get(col, (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        out = (s - lo) / (hi - lo)
        return out.clip(0.0, 1.0).fillna(default)

    @staticmethod
    def bounded01(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.dropna().empty:
            return pd.Series(default, index=df.index, dtype=float)
        mx = s.quantile(0.99)
        if mx > 1.5:
            s = s / 100.0
        return s.clip(0.0, 1.0).fillna(default)

def triangular_mid(x: pd.Series, center: float = 0.5, width: float = 0.35) -> pd.Series:
    return (1.0 - (x - center).abs() / max(width, 1e-9)).clip(0.0, 1.0)
