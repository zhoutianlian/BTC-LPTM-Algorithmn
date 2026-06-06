from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .data_loader import read_csv_normalized
from .utils import ensure_datetime, setup_logger, normalize_columns, normalize_col
from .validation import (
    FeatureContractError,
    NoLeakageReport,
    find_forbidden_columns,
    validate_available_lte_time,
    validate_required_columns,
)


class CanonicalFrameBuilder:
    """Build a causal, source-clock canonical feature frame.

    v2 supports the new upstream feature contract:
      * path_absorption_multiscale.csv with 6h/12h/24h/48h path context/label/scores.
      * price_context_features.csv with past-only price/volatility/range/trend/jump context.

    All joins are exact on source-clock where possible or backward-asof with a bounded tolerance.
    Future-diagnostic columns remain blacklisted.
    """

    def __init__(self, feature_contract: dict[str, Any]):
        self.contract = feature_contract
        self.logger = setup_logger()

    def _load_required(self, paths: dict[str, Path | None], role: str) -> pd.DataFrame:
        path = paths.get(role)
        if path is None:
            raise FeatureContractError(f"Required file role missing: {role}")
        return read_csv_normalized(path)

    def _load_optional(self, paths: dict[str, Path | None], role: str) -> pd.DataFrame | None:
        path = paths.get(role)
        if path is None:
            return None
        return read_csv_normalized(path)

    def _read_optional_selected(self, path: Path, role: str, fallback_all: bool = False) -> pd.DataFrame:
        """Read a CSV role with contract-driven selected columns to control memory.

        Raw columns in current upstream files are already snake_case, but this function also
        tolerates minor naming differences through normalize_col.
        """
        required = self.contract.get("required_core_columns", {}).get(role, [])
        optional = self.contract.get("optional_columns", {}).get(role, [])
        wanted = {normalize_col(c) for c in (["time", "available_time", "price_feature_time", "price_feature_age_min", "split"] + required + optional)}
        # Always keep engineered groups that are useful and low risk for this project.
        if role == "path_absorption_multiscale":
            wanted_prefixes = ("path_", "e_", "score_", "state_proxy")
        elif role == "price_context_features":
            wanted_prefixes = ("past_return_", "realized_vol_", "range_", "trend_", "bar_direction_", "block_direction_", "vol_of_vol", "jump_", "max_jump_", "signed_max_jump", "price_")
        else:
            wanted_prefixes = tuple()

        header = pd.read_csv(path, nrows=0)
        usecols = []
        for raw in header.columns:
            norm = normalize_col(raw)
            if norm in wanted or any(norm.startswith(pref) for pref in wanted_prefixes):
                usecols.append(raw)
        if not usecols and fallback_all:
            return read_csv_normalized(path)
        df = pd.read_csv(path, usecols=usecols)
        return normalize_columns(df)

    def _check_required(self, role: str, df: pd.DataFrame, missing: dict[str, list[str]]) -> None:
        req = self.contract.get("required_core_columns", {}).get(role, [])
        m = validate_required_columns(df, role, req)
        if m:
            missing[role] = m

    @staticmethod
    def _dedup_time(df: pd.DataFrame, role: str) -> pd.DataFrame:
        if "time" not in df.columns:
            return df
        n_before = len(df)
        df = df.sort_values("time").drop_duplicates("time", keep="last")
        n_after = len(df)
        if n_before != n_after:
            setup_logger().warning("%s: dropped %d duplicate time rows", role, n_before - n_after)
        return df

    @staticmethod
    def _merge_exact_coalesce(left: pd.DataFrame, right: pd.DataFrame, on: str = "time", prefer_right: bool = True) -> pd.DataFrame:
        right = right.copy()
        for c in list(right.columns):
            if c != on and c in left.columns:
                right = right.rename(columns={c: f"{c}__r"})
        out = left.merge(right, on=on, how="left")
        for c in list(out.columns):
            if c.endswith("__r"):
                base = c[:-3]
                if base in out.columns:
                    out[base] = out[c].combine_first(out[base]) if prefer_right else out[base].combine_first(out[c])
                else:
                    out[base] = out[c]
                out = out.drop(columns=[c])
        return out

    @staticmethod
    def _merge_asof_coalesce(
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: str = "time",
        tolerance: pd.Timedelta | None = None,
        prefer_right: bool = True,
    ) -> pd.DataFrame:
        right = right.copy()
        for c in list(right.columns):
            if c != on and c in left.columns:
                right = right.rename(columns={c: f"{c}__r"})
        out = pd.merge_asof(
            left.sort_values(on),
            right.sort_values(on),
            on=on,
            direction="backward",
            tolerance=tolerance,
        )
        for c in list(out.columns):
            if c.endswith("__r"):
                base = c[:-3]
                if base in out.columns:
                    out[base] = out[c].combine_first(out[base]) if prefer_right else out[base].combine_first(out[c])
                else:
                    out[base] = out[c]
                out = out.drop(columns=[c])
        return out

    def _select_columns(self, df: pd.DataFrame, base_cols: list[str], optional_cols: list[str] | None = None) -> pd.DataFrame:
        cols = []
        for c in base_cols + (optional_cols or []):
            if c in df.columns and c not in cols:
                cols.append(c)
        return df[cols].copy()

    def build(self, paths: dict[str, Path | None]) -> tuple[pd.DataFrame, NoLeakageReport]:
        missing_required: dict[str, list[str]] = {}
        available_violations: dict[str, int] = {}
        notes: list[str] = []

        # 1) Primary source clock from base_context.
        base = self._load_required(paths, "base_context")
        self._check_required("base_context", base, missing_required)
        base = ensure_datetime(base, ["time", "liq_feature_time", "liq_feature_time_raw"])
        base = base.sort_values("time")
        duplicate_time_count = int(base["time"].duplicated().sum()) if "time" in base.columns else 0
        base = self._dedup_time(base, "base_context")
        if "time" not in base.columns:
            raise FeatureContractError("base_context.csv must contain time column")
        canonical = base.copy()
        canonical["feature_version"] = self.contract.get("version", "unknown")
        canonical["source_clock_id"] = np.arange(len(canonical), dtype=int)

        # 2) PLIE/HMM exact source-clock join. Exclude future diagnostic columns by construction.
        plie = self._load_required(paths, "plie")
        self._check_required("plie", plie, missing_required)
        forbidden = find_forbidden_columns(plie.columns, self.contract.get("no_future_columns_regex", []))
        if forbidden:
            notes.append(f"Raw PLIE file contains forbidden columns and they were excluded: {forbidden}")
        live_plie_cols = [
            "time", "p_state_1", "p_state_2", "p_state_3", "p_state_4", "p_state_5",
            "hmm_state", "hmm_conf", "liq_entropy", "age_in_state_source",
            "state_pressure_direction", "state_severity",
            "plie_main_bps", "plie_abs_main_bps",
            "plie_passive_20m_bps", "plie_passive_30m_bps", "plie_passive_60m_bps",
            "plie_direction", "plie_reliability", "plie_intensity", "plie_accel_pos",
            "plie_phase", "plie_strong_entry", "plie_transition_type",
            "plie_transition_severity", "plie_strong_state",
        ]
        plie = ensure_datetime(plie, ["time", "liq_feature_time", "liq_feature_time_raw"])
        plie = self._dedup_time(plie, "plie")
        plie_sel = plie[[c for c in live_plie_cols if c in plie.columns]].copy()
        canonical = self._merge_exact_coalesce(canonical, plie_sel, on="time", prefer_right=True)

        # 3) Matured market response memory. Do not use raw event-level future response directly.
        mem = self._load_required(paths, "absorption_memory")
        self._check_required("absorption_memory", mem, missing_required)
        mem = ensure_datetime(mem, ["time", "latest_abs_available_time_30m", "last_directional_core_available_time_30m"])
        mem = self._dedup_time(mem, "absorption_memory")
        mem_cols = ["time"] + [
            c for c in mem.columns
            if c.startswith("mar_") or c in {"latest_abs_available_time_30m", "last_directional_core_available_time_30m"}
        ]
        mem_sel = mem[mem_cols].copy()
        available_violations["absorption_memory_latest_abs_available_time_30m"] = validate_available_lte_time(
            mem_sel, "latest_abs_available_time_30m"
        )
        available_violations["absorption_memory_last_directional_core_available_time_30m"] = validate_available_lte_time(
            mem_sel, "last_directional_core_available_time_30m"
        )
        canonical = self._merge_exact_coalesce(canonical, mem_sel, on="time", prefer_right=True)

        # 4) New multiscale path absorption preferred when available.
        multi = None
        if paths.get("path_absorption_multiscale"):
            multi = self._read_optional_selected(paths["path_absorption_multiscale"], "path_absorption_multiscale")
        if multi is not None:
            self._check_required("path_absorption_multiscale", multi, missing_required)
            multi = ensure_datetime(multi, ["time", "available_time"])
            multi = self._dedup_time(multi, "path_absorption_multiscale")
            available_violations["path_absorption_multiscale_available_time"] = validate_available_lte_time(multi, "available_time")
            rename = {"available_time": "path_multiscale_available_time"}
            if "split" in multi.columns:
                rename["split"] = "path_multiscale_split"
            multi = multi.rename(columns=rename)
            keep = [c for c in multi.columns if c == "time" or c.startswith("path_") or c.startswith("e_") or c.startswith("score_") or c in {"state_proxy_label", "state_proxy_margin"}]
            multi_sel = multi[keep].copy()
            canonical = self._merge_exact_coalesce(canonical, multi_sel, on="time", prefer_right=True)
            notes.append("Using path_absorption_multiscale.csv as preferred 6h/12h/24h/48h path layer.")
        else:
            notes.append("path_absorption_multiscale.csv missing; falling back to 24h path_absorption.csv only.")

        # 5) Legacy long path table still included for backward compatibility and for path_available_time_24h.
        path = self._load_required(paths, "path_absorption")
        self._check_required("path_absorption", path, missing_required)
        path = ensure_datetime(path, ["time", "available_time"])
        if "window_hours" in path.columns:
            unique_windows = sorted(pd.Series(path["window_hours"]).dropna().unique().tolist())
            notes.append(f"path_absorption available window_hours={unique_windows}; canonical frame uses 24h compatibility fields.")
            if 24 in unique_windows:
                path = path[path["window_hours"] == 24].copy()
        path = self._dedup_time(path, "path_absorption")
        available_violations["path_absorption_available_time"] = validate_available_lte_time(path, "available_time")
        path_rename = {
            "available_time": "path_available_time_24h",
            "window_hours": "path_window_hours_24h",
            "path_context": "path_context_24h",
            "path_label": "path_label_24h",
            "path_absorption_score_0_100": "path_absorption_score_24h",
            "path_pressure_rejection_score": "path_pressure_rejection_score_24h",
            "path_active_dominance_score": "path_active_dominance_score_24h",
            "path_active_dominance_price_score": "path_active_dominance_price_score_24h",
            "path_transmission_ratio": "path_transmission_ratio_24h",
            "path_direction_consistency": "path_direction_consistency_24h",
            "path_quality": "path_quality_24h",
            "path_data_quality": "path_data_quality_24h",
            "path_signal_clarity": "path_signal_clarity_24h",
            "path_activity_level": "path_activity_level_24h",
            "path_cascade_score": "path_cascade_score_24h",
            "path_return_bps": "path_return_bps_24h",
            "path_snr": "path_snr_24h",
            "path_active_z": "path_active_z_24h",
            "path_liq_neutrality_score": "path_liq_neutrality_score_24h",
            "path_quiet_score": "path_quiet_score_24h",
            "path_chop_score": "path_chop_score_24h",
        }
        path_cols = ["time"] + [c for c in path_rename if c in path.columns]
        path_sel = path[path_cols].rename(columns=path_rename)
        canonical = self._merge_exact_coalesce(canonical, path_sel, on="time", prefer_right=False)

        # 6) New price context features, backward-asof joined, preferred over base columns when present.
        price_ctx_path = paths.get("price_context_features")
        if price_ctx_path:
            price_ctx = self._read_optional_selected(price_ctx_path, "price_context_features")
            self._check_required("price_context_features", price_ctx, missing_required)
            price_ctx = ensure_datetime(price_ctx, ["time", "price_feature_time"])
            price_ctx = self._dedup_time(price_ctx, "price_context_features")
            price_ctx["price_context_join_time"] = price_ctx["time"]
            # Keep all engineered price context fields plus feature-time metadata. Do not include raw OHLC if present.
            keep = ["time", "price_context_join_time", "price_feature_time", "price_feature_age_min"] + [
                c for c in price_ctx.columns
                if c not in {"time", "price_context_join_time", "price_feature_time", "price_feature_age_min"}
                and (
                    c.startswith("past_return_") or c.startswith("realized_vol_") or
                    c.startswith("range_") or c.startswith("trend_") or
                    c.startswith("bar_direction_") or c.startswith("block_direction_") or
                    c.startswith("vol_of_vol") or c.startswith("jump_") or c.startswith("max_jump_") or
                    c.startswith("signed_max_jump") or c.startswith("price_")
                )
            ]
            keep = [c for i, c in enumerate(keep) if c in price_ctx.columns and c not in keep[:i]]
            price_ctx = price_ctx[keep].copy()
            tol = pd.Timedelta(minutes=int(self.contract.get("join", {}).get("price_context_asof_tolerance_minutes", self.contract.get("join", {}).get("asof_tolerance_minutes", 70))))
            canonical = self._merge_asof_coalesce(canonical, price_ctx, on="time", tolerance=tol, prefer_right=True)
            canonical["price_context_lag_min"] = (
                (canonical["time"] - canonical["price_context_join_time"]).dt.total_seconds() / 60.0
                if "price_context_join_time" in canonical.columns else np.nan
            )
            notes.append("Using price_context_features.csv as preferred past-only price context layer.")
        else:
            notes.append("price_context_features.csv missing; price context will rely on base_context and liqprice fallback.")

        # 7) Legacy liqprice proxy as optional fallback/auxiliary.
        liqprice_path = paths.get("liqprice_features")
        if liqprice_path:
            liqprice = read_csv_normalized(liqprice_path)
            liqprice = ensure_datetime(liqprice, ["time"])
            liqprice = self._dedup_time(liqprice, "liqprice_features")
            liqprice = liqprice.sort_values("time")
            liqprice["liqprice_feature_time"] = liqprice["time"]
            liq_cols = [
                "time", "liqprice_feature_time",
                "fll_spike_kama", "fsl_spike_kama",
                "fll_velocity_gaussian", "fll_acceleration_gaussian",
                "fsl_velocity_gaussian", "fsl_acceleration_gaussian",
                "trend_pressure", "kalman_slope", "vol_adaptive",
            ]
            liqprice = liqprice[[c for c in liq_cols if c in liqprice.columns]]
            tol = pd.Timedelta(minutes=int(self.contract.get("join", {}).get("asof_tolerance_minutes", 70)))
            canonical = self._merge_asof_coalesce(canonical, liqprice, on="time", tolerance=tol, prefer_right=False)
            if "liqprice_feature_time" in canonical.columns:
                canonical["liqprice_context_lag_min"] = (canonical["time"] - canonical["liqprice_feature_time"]).dt.total_seconds() / 60.0
        else:
            notes.append("liqprice_features.csv missing; old price/vol proxy unavailable.")

        # 8) Optional FHMV/liquidation stress auxiliary features.
        fhmv_path = paths.get("fhmv_liq_features")
        if fhmv_path:
            fhmv = read_csv_normalized(fhmv_path)
            fhmv = ensure_datetime(fhmv, ["time"])
            fhmv = self._dedup_time(fhmv, "fhmv_liq_features")
            rename = {c: f"fhmv_{c}" for c in fhmv.columns if c not in {"time", "price"}}
            fhmv = fhmv.rename(columns=rename)
            fhmv["fhmv_feature_time"] = fhmv["time"]
            cols = ["time", "fhmv_feature_time"] + list(rename.values())
            tol = pd.Timedelta(minutes=int(self.contract.get("join", {}).get("asof_tolerance_minutes", 70)))
            canonical = self._merge_asof_coalesce(canonical, fhmv[cols], on="time", tolerance=tol, prefer_right=True)
            if "fhmv_feature_time" in canonical.columns:
                canonical["fhmv_context_lag_min"] = (canonical["time"] - canonical["fhmv_feature_time"]).dt.total_seconds() / 60.0

        # 9) Derived quality/staleness flags. These are protective evidence, not market states by themselves.
        max_liq_age = float(self.contract.get("join", {}).get("max_liq_feature_age_minutes", 90))
        if "liq_feature_age_min" in canonical.columns:
            canonical["source_stale_flag"] = (pd.to_numeric(canonical["liq_feature_age_min"], errors="coerce") > max_liq_age).fillna(True).astype(int)
        else:
            canonical["source_stale_flag"] = 1

        if "mar_directional_core_age_hours_30m" in canonical.columns:
            max_mar_age = float(self.contract.get("join", {}).get("max_market_response_age_hours", 12))
            canonical["market_response_stale_flag"] = (
                pd.to_numeric(canonical["mar_directional_core_age_hours_30m"], errors="coerce") > max_mar_age
            ).fillna(False).astype(int)
        else:
            canonical["market_response_stale_flag"] = 1

        if "path_multiscale_available_time" in canonical.columns:
            canonical["path_stale_flag"] = canonical["path_multiscale_available_time"].isna().astype(int)
        elif "path_available_time_24h" in canonical.columns:
            canonical["path_stale_flag"] = canonical["path_available_time_24h"].isna().astype(int)
        else:
            canonical["path_stale_flag"] = 1

        if "price_context_lag_min" in canonical.columns:
            tol_min = float(self.contract.get("join", {}).get("price_context_asof_tolerance_minutes", self.contract.get("join", {}).get("asof_tolerance_minutes", 70)))
            canonical["price_context_stale_flag"] = (canonical["price_context_lag_min"].isna() | (canonical["price_context_lag_min"] > tol_min)).astype(int)
        elif "liqprice_context_lag_min" in canonical.columns:
            tol_min = float(self.contract.get("join", {}).get("asof_tolerance_minutes", 70))
            canonical["price_context_stale_flag"] = (canonical["liqprice_context_lag_min"].isna() | (canonical["liqprice_context_lag_min"] > tol_min)).astype(int)
        else:
            canonical["price_context_stale_flag"] = 1

        # Price data quality flags aggregate the explicit price context flags across core horizons.
        price_gap_cols = [c for c in ["price_gap_flag_1h", "price_gap_flag_6h", "price_gap_flag_24h"] if c in canonical.columns]
        price_missing_cols = [c for c in ["price_missing_ratio_1h", "price_missing_ratio_6h", "price_missing_ratio_24h"] if c in canonical.columns]
        price_outlier_cols = [c for c in ["price_outlier_flag_1h", "price_outlier_flag_6h", "price_outlier_flag_24h"] if c in canonical.columns]
        canonical["price_gap_any_flag"] = canonical[price_gap_cols].max(axis=1).fillna(0).astype(int) if price_gap_cols else 0
        canonical["price_missing_rate_max"] = canonical[price_missing_cols].max(axis=1).fillna(0).clip(0, 1) if price_missing_cols else 0.0
        canonical["price_outlier_any_flag"] = canonical[price_outlier_cols].max(axis=1).fillna(0).astype(int) if price_outlier_cols else 0

        # Generic data gap flag: true structural data problems only, not low activity or quiet markets.
        canonical["data_gap_flag"] = (
            (canonical["source_stale_flag"] > 0)
            | (canonical["path_stale_flag"] > 0)
            | (canonical["price_context_stale_flag"] > 0)
            | (canonical["price_gap_any_flag"] > 0)
            | (canonical["price_missing_rate_max"] > 0.20)
        ).astype(int)

        core_check = [
            "plie_direction", "plie_reliability", "plie_intensity", "hmm_conf", "liq_entropy",
            "path_context_6h", "path_label_6h", "path_context_12h", "path_label_12h",
            "path_context_24h", "path_label_24h", "path_context_48h", "path_label_48h",
            "mar_response_conflict_score", "realized_vol_6h_bps", "trend_strength_6h", "range_compression_6h",
        ]
        present_core = [c for c in core_check if c in canonical.columns]
        canonical["missing_rate_row"] = canonical[present_core].isna().mean(axis=1) if present_core else 1.0
        canonical["critical_missing_flag"] = (canonical["missing_rate_row"] > 0.35).astype(int)
        canonical["sample_weight"] = 1.0

        forbidden_final = find_forbidden_columns(canonical.columns, self.contract.get("no_future_columns_regex", []))
        passed = not forbidden_final and not missing_required and all(v == 0 for v in available_violations.values())
        report = NoLeakageReport(
            passed=passed,
            forbidden_columns_found=forbidden_final,
            available_time_violations=available_violations,
            missing_required_columns=missing_required,
            duplicate_time_count=duplicate_time_count,
            notes=notes,
        )
        if forbidden_final:
            raise FeatureContractError(f"Forbidden future columns entered canonical frame: {forbidden_final}")
        if missing_required:
            raise FeatureContractError(f"Missing required columns: {missing_required}")
        canonical = canonical.sort_values("time").reset_index(drop=True)
        return canonical, report
