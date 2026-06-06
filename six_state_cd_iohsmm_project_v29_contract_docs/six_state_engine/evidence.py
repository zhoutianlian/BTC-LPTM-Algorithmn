from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from .constants import (
    STATES, DIRECTIONAL_CONTEXTS, NEUTRAL_CONTEXTS, MIXED_CONTEXTS,
    CASCADE_LABELS, BASELINE_LABELS, REJECTION_LABELS, ACTIVE_LABELS,
    QUIET_LABELS, MIXED_CHOP_LABELS,
)
from .transforms import QuantileScaler, triangular_mid
from .utils import softmax_matrix


class EvidenceBuilder:
    """Build semantically anchored evidence primitives for SSEM/CD-IOHSMM-lite.

    v2 consumes multiscale path absorption and the new price_context layer. The goal is not
    price prediction; price context is used only to separate quiet consolidation, active volatile
    trends, jump-driven instability, and orderly trends.
    """

    PATH_WINDOWS = ["6h", "12h", "24h", "48h"]
    # Fast windows matter more for transitions and RC/VT, but 24h/48h still provide slow context.
    PATH_WEIGHTS = {"6h": 0.35, "12h": 0.30, "24h": 0.20, "48h": 0.15}

    def __init__(self, model_config: dict[str, Any]):
        ev_cfg = model_config.get("evidence", {})
        self.model_config = model_config
        self.scaler = QuantileScaler(
            q_low=float(ev_cfg.get("robust_quantile_low", 0.05)),
            q_high=float(ev_cfg.get("robust_quantile_high", 0.95)),
        )

    def fit(self, df: pd.DataFrame) -> "EvidenceBuilder":
        scale_cols = [
            # PLIE/HMM current pressure
            "plie_intensity", "state_severity", "plie_accel_pos",
            # Market response memory
            "mar_amplification_persistence_6_30m", "mar_absorption_persistence_6_30m",
            "mar_abs_score_q_staleaware_ewm_6_30m", "mar_takeover_count_12_30m",
            "mar_neutral_active_strength_evidence_ewm_6_30m", "mar_response_conflict_score",
            "liq_entropy", "hmm_conf",
            # Price context / old proxies
            "realized_vol_1h_bps", "realized_vol_6h_bps", "realized_vol_24h_bps",
            "vol_of_vol_6h", "vol_of_vol_24h", "vol_of_vol_48h",
            "max_jump_z_1h", "max_jump_z_6h", "max_jump_z_24h",
            "jump_count_1h", "jump_count_6h", "jump_count_24h",
            "trend_pressure", "kalman_slope", "vol_adaptive",
            "fll_spike_kama", "fsl_spike_kama", "fll_acceleration_gaussian", "fsl_acceleration_gaussian",
        ]
        for w in self.PATH_WINDOWS:
            scale_cols.extend([
                f"path_cascade_score_{w}", f"path_absorption_score_{w}", f"path_pressure_rejection_score_{w}",
                f"path_active_dominance_score_{w}", f"path_active_dominance_price_score_{w}",
                f"path_transmission_ratio_{w}", f"path_direction_consistency_{w}", f"path_activity_level_{w}",
                f"path_pressure_mass_{w}", f"path_directionality_{w}",
            ])
        if "split" in df.columns:
            train_mask = df["split"].astype(str).str.lower().eq("train")
            if train_mask.sum() < 100:
                train_mask = None
        else:
            train_mask = pd.Series(np.arange(len(df)) < int(len(df) * 0.7), index=df.index)
        self.scaler.fit(df, scale_cols, train_mask=train_mask)
        return self

    @staticmethod
    def _series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    @staticmethod
    def _cat(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series("", index=df.index)
        return df[col].astype(str).str.lower()

    @staticmethod
    def _clip01(x: pd.Series | np.ndarray | float) -> pd.Series:
        return pd.Series(x).clip(0, 1) if not isinstance(x, pd.Series) else x.clip(0, 1)

    def _weighted_label(self, df: pd.DataFrame, prefix: str, values: set[str], default_window: str = "24h") -> pd.Series:
        idx = df.index
        total = pd.Series(0.0, index=idx)
        wsum = 0.0
        any_present = False
        for w, wt in self.PATH_WEIGHTS.items():
            col = f"{prefix}_{w}"
            if col in df.columns:
                any_present = True
                total = total + wt * self._cat(df, col).isin(values).astype(float)
                wsum += wt
        if any_present and wsum > 0:
            return (total / wsum).clip(0, 1)
        col = f"{prefix}_{default_window}"
        if col in df.columns:
            return self._cat(df, col).isin(values).astype(float)
        return pd.Series(0.0, index=idx)

    def _weighted_context(self, df: pd.DataFrame, values: set[str]) -> pd.Series:
        return self._weighted_label(df, "path_context", values)

    def _weighted_score(self, df: pd.DataFrame, base: str, bounded: bool = True, scaled: bool = False, default: float = 0.0) -> pd.Series:
        idx = df.index
        total = pd.Series(0.0, index=idx)
        wsum = 0.0
        for w, wt in self.PATH_WEIGHTS.items():
            col = f"{base}_{w}"
            if col not in df.columns:
                continue
            if bounded:
                x = self.scaler.bounded01(df, col, default=default)
            elif scaled:
                x = self.scaler.transform01(df, col, default=default)
            else:
                x = self._series(df, col, default=default)
            total = total + wt * x
            wsum += wt
        if wsum > 0:
            return (total / wsum).clip(0, 1)
        col = f"{base}_24h"
        if col in df.columns:
            return self.scaler.bounded01(df, col, default=default) if bounded else self.scaler.transform01(df, col, default=default)
        return pd.Series(default, index=idx).clip(0, 1)

    def _weighted_quality(self, df: pd.DataFrame, base: str, default: float = 1.0) -> pd.Series:
        return self._weighted_score(df, base, bounded=True, scaled=False, default=default)

    def _cross_window_conflict(self, df: pd.DataFrame) -> pd.Series:
        idx = df.index
        # Prefer upstream audit field if available.
        if "e_amb_cross_window_conflict" in df.columns:
            return self.scaler.bounded01(df, "e_amb_cross_window_conflict", 0.0)
        ctx6 = self._cat(df, "path_context_6h")
        ctx24 = self._cat(df, "path_context_24h")
        lab6 = self._cat(df, "path_label_6h")
        lab24 = self._cat(df, "path_label_24h")
        ctx_conf = (
            (ctx6.isin(NEUTRAL_CONTEXTS) & ctx24.isin(DIRECTIONAL_CONTEXTS)) |
            (ctx6.isin(DIRECTIONAL_CONTEXTS) & ctx24.isin(NEUTRAL_CONTEXTS)) |
            (ctx6.isin(MIXED_CONTEXTS) & ~ctx24.isin(MIXED_CONTEXTS)) |
            (~ctx6.isin(MIXED_CONTEXTS) & ctx24.isin(MIXED_CONTEXTS))
        ).astype(float)
        lab_conf = (
            (lab6.isin(QUIET_LABELS) & lab24.isin(REJECTION_LABELS | CASCADE_LABELS)) |
            (lab6.isin(ACTIVE_LABELS) & lab24.isin(REJECTION_LABELS | CASCADE_LABELS)) |
            (lab6.isin(CASCADE_LABELS) & lab24.isin(REJECTION_LABELS)) |
            (lab6.isin(REJECTION_LABELS) & lab24.isin(CASCADE_LABELS))
        ).astype(float)
        return (0.55 * ctx_conf + 0.45 * lab_conf).clip(0, 1).fillna(0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        idx = out.index
        s01 = self.scaler.transform01
        b01 = self.scaler.bounded01

        # ----- Path pressure context and labels, multiscale -----
        e_directional_context = self._weighted_context(out, DIRECTIONAL_CONTEXTS)
        e_neutral_context = self._weighted_context(out, NEUTRAL_CONTEXTS)
        e_mixed_context = self._weighted_context(out, MIXED_CONTEXTS)
        e_neutral_or_mixed = ((e_neutral_context + e_mixed_context).clip(0, 1))

        label_cascade = self._weighted_label(out, "path_label", CASCADE_LABELS)
        label_baseline_only = self._weighted_label(out, "path_label", {"path_baseline_transmission"})
        label_partial = self._weighted_label(out, "path_label", {"path_partial_absorption"})
        label_full_stall = self._weighted_label(out, "path_label", {"path_full_absorption_stall"})
        label_pressure_rejection = self._weighted_label(out, "path_label", {"path_pressure_rejection"})
        label_reversal_takeover = self._weighted_label(out, "path_label", {"path_reversal_takeover"})
        label_rejection = (label_full_stall * 0.65 + label_pressure_rejection * 0.90 + label_reversal_takeover * 1.0).clip(0, 1)
        label_active = self._weighted_label(out, "path_label", ACTIVE_LABELS)
        label_quiet = self._weighted_label(out, "path_label", QUIET_LABELS)
        label_mixed_chop = self._weighted_label(out, "path_label", MIXED_CHOP_LABELS)

        path_cascade_score = self._weighted_score(out, "path_cascade_score", bounded=True)
        path_absorption_score = self._weighted_score(out, "path_absorption_score", bounded=True)
        path_rejection_score = self._weighted_score(out, "path_pressure_rejection_score", bounded=True)
        path_active_score = self._weighted_score(out, "path_active_dominance_score", bounded=True)
        path_active_price_score = self._weighted_score(out, "path_active_dominance_price_score", bounded=True)
        path_data_quality = self._weighted_quality(out, "path_data_quality", default=1.0)
        path_signal_clarity = self._weighted_quality(out, "path_signal_clarity", default=0.5)
        path_activity_level = self._weighted_quality(out, "path_activity_level", default=0.4)

        # ----- Current PLIE/HMM pressure -----
        plie_reliability = b01(out, "plie_reliability", 0.0)
        plie_direction_abs = self._series(out, "plie_direction", 0).abs().clip(0, 1)
        e_directional_pressure = (e_directional_context * (0.35 + 0.65 * plie_reliability) * plie_direction_abs).clip(0, 1)

        plie_intensity = s01(out, "plie_intensity")
        state_severity = s01(out, "state_severity")
        e_pressure_strength = (0.50 * plie_intensity + 0.35 * state_severity + 0.15 * plie_reliability).clip(0, 1)
        e_pressure_strength = (e_pressure_strength * (0.35 + 0.65 * plie_direction_abs)).clip(0, 1)
        e_accel = s01(out, "plie_accel_pos")
        e_strong_entry = self._series(out, "plie_strong_entry", 0).clip(0, 1)

        # ----- Price context -----
        rv_1h = s01(out, "realized_vol_1h_bps")
        rv_6h = s01(out, "realized_vol_6h_bps")
        rv_24h = s01(out, "realized_vol_24h_bps")
        jump_1h = b01(out, "jump_proxy_1h", 0.0)
        jump_6h = b01(out, "jump_proxy_6h", 0.0)
        jump_24h = b01(out, "jump_proxy_24h", 0.0)
        vov_6h = s01(out, "vol_of_vol_6h")
        vov_24h = s01(out, "vol_of_vol_24h")
        trend_strength_1h = b01(out, "trend_strength_1h", 0.0)
        trend_strength_6h = b01(out, "trend_strength_6h", 0.0)
        trend_strength_24h = b01(out, "trend_strength_24h", 0.0)
        trend_consistency_6h = b01(out, "trend_consistency_6h", 0.5)
        trend_consistency_24h = b01(out, "trend_consistency_24h", 0.5)
        range_compression_1h = b01(out, "range_compression_1h", 0.5)
        range_compression_6h = b01(out, "range_compression_6h", 0.5)
        range_compression_24h = b01(out, "range_compression_24h", 0.5)

        # Short-horizon responsiveness signals. These are strictly past-only price-context
        # features. They do not redefine the six states; they provide a controlled fast lane
        # for 1h shocks and an input to state-exit pressure.
        resp_cfg = self.model_config.get("responsiveness", {})
        shock_cfg = resp_cfg.get("price_shock", {}) if isinstance(resp_cfg, dict) else {}
        max_missing_1h = float(shock_cfg.get("max_price_missing_ratio_1h", 0.20))
        block_on_price_gap = bool(shock_cfg.get("block_on_price_gap", True))

        max_jump_z_1h = s01(out, "max_jump_z_1h")
        price_missing_ratio_1h = self._series(out, "price_missing_ratio_1h", 0.0).clip(0, 1)
        price_gap_flag_1h = self._series(out, "price_gap_flag_1h", 0.0).clip(0, 1)
        trend_direction_1h = self._series(out, "trend_direction_1h", 0.0).clip(-1, 1)
        plie_direction_raw = self._series(out, "plie_direction", 0.0).clip(-1, 1)

        e_price_shock_1h_raw = pd.concat([rv_1h, max_jump_z_1h, jump_1h], axis=1).max(axis=1).clip(0, 1)
        e_shock_data_valid = (price_missing_ratio_1h <= max_missing_1h).astype(float)
        if block_on_price_gap:
            e_shock_data_valid = e_shock_data_valid * (price_gap_flag_1h <= 0.5).astype(float)
        e_short_impulse_1h = (e_price_shock_1h_raw * trend_strength_1h * e_shock_data_valid).clip(0, 1)
        plie_has_direction = (plie_direction_raw.abs() > 0.5).astype(float)
        trend_has_direction = (trend_direction_1h.abs() > 0.5).astype(float)
        aligned = ((trend_direction_1h * plie_direction_raw) > 0).astype(float)
        against = ((trend_direction_1h * plie_direction_raw) < 0).astype(float)
        e_shock_aligned_plie = (e_short_impulse_1h * plie_has_direction * trend_has_direction * aligned).clip(0, 1)
        e_shock_against_plie = (e_short_impulse_1h * plie_has_direction * trend_has_direction * against).clip(0, 1)
        e_shock_active_breakout = (e_short_impulse_1h * e_neutral_or_mixed * (1 - e_directional_pressure).clip(0, 1)).clip(0, 1)

        # Composite high-volatility/jump/trend signals. Legacy proxies only add weak support.
        legacy_vol = pd.concat([
            s01(out, "vol_adaptive"), s01(out, "trend_pressure", abs_value=True), s01(out, "kalman_slope", abs_value=True),
            s01(out, "fll_spike_kama", abs_value=True), s01(out, "fsl_spike_kama", abs_value=True)
        ], axis=1).mean(axis=1)
        e_realized_vol = (0.20 * rv_1h + 0.45 * rv_6h + 0.35 * rv_24h).clip(0, 1)
        e_jump = (0.30 * jump_1h + 0.45 * jump_6h + 0.25 * jump_24h).clip(0, 1)
        e_vol_of_vol = (0.55 * vov_6h + 0.45 * vov_24h).clip(0, 1)
        e_trend_strength = (0.25 * trend_strength_1h + 0.45 * trend_strength_6h + 0.30 * trend_strength_24h).clip(0, 1)
        e_trend_consistency = (0.55 * trend_consistency_6h + 0.45 * trend_consistency_24h).clip(0, 1)
        e_range_compression = (0.20 * range_compression_1h + 0.45 * range_compression_6h + 0.35 * range_compression_24h).clip(0, 1)
        e_volatility = (0.45 * e_realized_vol + 0.20 * e_jump + 0.15 * e_vol_of_vol + 0.15 * e_trend_strength + 0.05 * legacy_vol).clip(0, 1)
        e_low_vol = (1 - e_realized_vol).clip(0, 1)
        e_low_trend = (1 - e_trend_strength).clip(0, 1)
        e_low_jump = (1 - e_jump).clip(0, 1)

        # ----- State-mechanism evidence -----
        e_cascade = (
            0.38 * label_cascade
            + 0.26 * path_cascade_score
            + 0.16 * s01(out, "mar_amplification_persistence_6_30m")
            + 0.10 * e_accel
            + 0.06 * e_jump
            + 0.04 * e_strong_entry
        ).clip(0, 1)

        # Partial absorption is orderly/transitional, not strong RHA.
        e_baseline = (0.58 * label_baseline_only + 0.28 * label_partial + 0.14 * e_directional_context).clip(0, 1)

        e_absorption = (
            0.45 * path_absorption_score
            + 0.25 * s01(out, "mar_absorption_persistence_6_30m")
            + 0.20 * s01(out, "mar_abs_score_q_staleaware_ewm_6_30m")
            + 0.10 * label_full_stall
        ).clip(0, 1)

        e_rejection = (
            0.36 * path_rejection_score
            + 0.20 * label_pressure_rejection
            + 0.22 * label_reversal_takeover
            + 0.10 * label_full_stall
            + 0.12 * s01(out, "mar_takeover_count_12_30m")
        ).clip(0, 1)
        e_rha_strict = (e_directional_pressure * (0.45 * e_rejection + 0.30 * e_absorption + 0.25 * (label_pressure_rejection + label_reversal_takeover).clip(0, 1))).clip(0, 1)

        e_active = (
            0.34 * path_active_score
            + 0.16 * path_active_price_score
            + 0.24 * label_active
            + 0.12 * e_trend_strength
            + 0.08 * s01(out, "mar_neutral_active_strength_evidence_ewm_6_30m")
            + 0.06 * e_neutral_or_mixed
        ).clip(0, 1)

        # Conflict is not low activity. It is contradictory evidence, mixed chop, ambiguity, or data problems.
        e_cross_window_conflict = self._cross_window_conflict(out)
        e_response_conflict = s01(out, "mar_response_conflict_score")
        e_label_margin_low = b01(out, "e_amb_label_margin_low", 0.0) if "e_amb_label_margin_low" in out.columns else pd.Series(0.0, index=idx)
        e_semantic_strength = pd.concat([e_baseline, e_cascade, e_rha_strict, e_active, label_quiet], axis=1).max(axis=1)
        # Cross-window disagreement is useful as uncertainty only when no strong semantic state
        # explains the disagreement. In directional pressure-rejection windows, treating all
        # short/long label disagreement as AMB would suppress the intended RHA state.
        e_conflict_pre = (
            0.38 * e_response_conflict
            + 0.20 * s01(out, "liq_entropy")
            + 0.10 * e_cross_window_conflict * (1 - 0.70 * e_rha_strict)
            + 0.16 * label_mixed_chop
            + 0.06 * e_label_margin_low * (1 - e_semantic_strength)
        ).clip(0, 1)

        hmm_conf = b01(out, "hmm_conf", 0.5)
        stale_flags = (
            self._series(out, "data_gap_flag", 0)
            + self._series(out, "source_stale_flag", 0)
            + self._series(out, "market_response_stale_flag", 0)
            + self._series(out, "path_stale_flag", 0)
            + self._series(out, "price_context_stale_flag", 0)
        ).clip(0, 5) / 5.0
        missing_rate = self._series(out, "missing_rate_row", 0).clip(0, 1)
        price_missing_max = self._series(out, "price_missing_rate_max", 0).clip(0, 1)
        price_gap_flag = self._series(out, "price_gap_any_flag", 0).clip(0, 1)
        # Use path_data_quality only; do not penalize quiet/low-activity markets as bad quality.
        e_quality_bad = (
            0.25 * (1 - hmm_conf)
            + 0.25 * (1 - path_data_quality)
            + 0.20 * missing_rate
            + 0.15 * stale_flags
            + 0.10 * price_missing_max
            + 0.05 * price_gap_flag
        ).clip(0, 1)

        e_conflict = (0.82 * e_conflict_pre + 0.18 * e_quality_bad).clip(0, 1)

        e_quiet_price = (0.34 * e_low_vol + 0.30 * e_range_compression + 0.24 * e_low_trend + 0.12 * e_low_jump).clip(0, 1)
        e_quiet = (0.55 * label_quiet + 0.45 * (e_neutral_context * e_quiet_price) - 0.35 * e_conflict).clip(0, 1)

        e_moderate_pressure = triangular_mid(e_pressure_strength, center=0.50, width=0.40)
        e_moderate_volatility = triangular_mid(e_volatility, center=0.45, width=0.38)
        e_orderly_trend = (0.45 * e_trend_strength + 0.35 * e_trend_consistency + 0.20 * e_moderate_volatility).clip(0, 1)

        evidence = pd.DataFrame(index=idx)
        # Context/pressure
        evidence["e_directional_context"] = e_directional_context
        evidence["e_neutral_context"] = e_neutral_context
        evidence["e_mixed_context"] = e_mixed_context
        evidence["e_neutral_or_mixed"] = e_neutral_or_mixed
        evidence["e_directional_pressure"] = e_directional_pressure
        evidence["e_pressure_strength"] = e_pressure_strength
        evidence["e_accel"] = e_accel
        evidence["e_strong_entry"] = e_strong_entry
        # Path-response mechanisms
        evidence["e_baseline"] = e_baseline
        evidence["e_partial_absorption"] = label_partial
        evidence["e_cascade"] = e_cascade
        evidence["e_absorption"] = e_absorption
        evidence["e_rejection"] = e_rejection
        evidence["e_rha_strict"] = e_rha_strict
        evidence["e_active"] = e_active
        evidence["e_quiet"] = e_quiet
        # Price context
        evidence["e_realized_vol"] = e_realized_vol
        evidence["e_volatility"] = e_volatility
        evidence["e_jump"] = e_jump
        evidence["e_vol_of_vol"] = e_vol_of_vol
        evidence["e_trend_strength"] = e_trend_strength
        evidence["e_trend_consistency"] = e_trend_consistency
        evidence["e_range_compression"] = e_range_compression
        evidence["e_low_vol"] = e_low_vol
        evidence["e_low_trend"] = e_low_trend
        evidence["e_low_jump"] = e_low_jump
        evidence["e_orderly_trend"] = e_orderly_trend
        # Short-horizon responsiveness / fast-lane diagnostics
        evidence["e_price_shock_1h"] = e_price_shock_1h_raw
        evidence["e_short_impulse_1h"] = e_short_impulse_1h
        evidence["e_shock_data_valid"] = e_shock_data_valid
        evidence["e_shock_aligned_plie"] = e_shock_aligned_plie
        evidence["e_shock_against_plie"] = e_shock_against_plie
        evidence["e_shock_active_breakout"] = e_shock_active_breakout
        # Quality/conflict
        evidence["e_path_data_quality"] = path_data_quality
        evidence["e_path_signal_clarity"] = path_signal_clarity
        evidence["e_path_activity_level"] = path_activity_level
        evidence["e_cross_window_conflict"] = e_cross_window_conflict
        evidence["e_label_margin_low"] = e_label_margin_low
        evidence["e_mixed_chop"] = label_mixed_chop
        evidence["e_conflict"] = e_conflict
        evidence["e_quality_bad"] = e_quality_bad
        evidence["e_moderate_pressure"] = e_moderate_pressure
        evidence["e_moderate_volatility"] = e_moderate_volatility

        scores = self.compute_scores(evidence)
        priors = softmax_matrix(
            scores[[f"score_{s}" for s in STATES]].to_numpy(float),
            temperature=float(self.model_config.get("model", {}).get("temperature", 1.35)),
            floor=float(self.model_config.get("model", {}).get("emission_floor", 1e-6)),
        )
        for i, state in enumerate(STATES):
            scores[f"prior_{state}"] = priors[:, i]

        return pd.concat([out, evidence, scores], axis=1)

    @staticmethod
    def compute_scores(e: pd.DataFrame) -> pd.DataFrame:
        score = pd.DataFrame(index=e.index)
        # v2.1: keep RC from becoming a generic low-vol bucket; strengthen ST/VT semantic anchors.
        score["score_ST"] = (
            1.95 * e["e_baseline"]
            + 0.85 * e["e_directional_pressure"]
            + 0.80 * e["e_orderly_trend"]
            + 0.45 * e["e_moderate_pressure"]
            + 0.40 * e["e_moderate_volatility"]
            - 0.60 * e["e_cascade"]
            - 0.45 * e["e_rejection"]
            - 0.75 * e["e_rha_strict"]
            - 0.95 * e["e_conflict"]
            - 0.35 * e["e_quality_bad"]
        )
        score["score_HPEM"] = (
            1.10 * e["e_directional_pressure"]
            + 1.00 * e["e_pressure_strength"]
            + 1.40 * e["e_cascade"]
            + 0.38 * e["e_volatility"]
            + 0.35 * e["e_jump"]
            + 0.18 * e["e_shock_aligned_plie"]
            + 0.30 * e["e_strong_entry"]
            - 0.80 * e["e_absorption"]
            - 0.90 * e["e_conflict"]
            - 0.35 * e["e_quality_bad"]
        )
        score["score_RHA"] = (
            1.15 * e["e_directional_pressure"]
            + 1.35 * e["e_rejection"]
            + 0.62 * e["e_absorption"]
            + 1.25 * e["e_rha_strict"]
            + 0.22 * e["e_shock_against_plie"]
            + 0.20 * e["e_pressure_strength"]
            - 0.60 * e["e_cascade"]
            - 1.10 * e["e_neutral_context"]
            - 0.40 * e["e_quality_bad"]
        )
        score["score_VT"] = (
            1.90 * e["e_active"]
            + 0.85 * e["e_volatility"]
            + 0.75 * e["e_trend_strength"]
            + 0.55 * e["e_neutral_or_mixed"]
            + 0.25 * e["e_jump"]
            + 0.20 * e["e_shock_active_breakout"]
            - 0.65 * e["e_directional_pressure"] * e["e_pressure_strength"]
            - 0.45 * e["e_cascade"]
            - 0.45 * e["e_conflict"]
        )
        score["score_RC"] = (
            2.00 * e["e_quiet"]
            + 0.55 * e["e_neutral_context"]
            + 0.35 * e["e_low_vol"]
            + 0.35 * e["e_range_compression"]
            + 0.25 * e["e_low_trend"]
            + 0.15 * e["e_low_jump"]
            - 0.75 * e["e_pressure_strength"]
            - 0.80 * e["e_active"]
            - 0.95 * e["e_conflict"]
            - 0.35 * e["e_quality_bad"]
        )
        score["score_AMB"] = (
            1.65 * e["e_conflict"]
            + 1.05 * e["e_quality_bad"]
            + 0.55 * e["e_mixed_context"]
            + 0.75 * e["e_mixed_chop"]
            + 0.25 * e["e_cross_window_conflict"] * (1 - 0.70 * e["e_rha_strict"])
            + 0.15 * e["e_label_margin_low"] * (1 - e[["e_baseline", "e_cascade", "e_rha_strict", "e_active", "e_quiet"]].max(axis=1))
            + 0.35 * (1 - e[["e_baseline", "e_cascade", "e_rejection", "e_active", "e_quiet"]].max(axis=1))
            - 0.45 * e["e_quiet"]
            - 0.30 * e["e_rha_strict"]
        )
        return score[[f"score_{s}" for s in STATES]]

    @staticmethod
    def top_evidence_for_state(row: pd.Series, state: str, n: int = 5) -> str:
        maps = {
            "ST": {
                "baseline/orderly transmission": row.get("e_baseline", 0),
                "orderly trend": row.get("e_orderly_trend", 0),
                "directional pressure": row.get("e_directional_pressure", 0),
                "moderate pressure": row.get("e_moderate_pressure", 0),
                "low conflict": 1 - row.get("e_conflict", 0),
            },
            "HPEM": {
                "cascade/amplification": row.get("e_cascade", 0),
                "strong liquidation pressure": row.get("e_pressure_strength", 0),
                "directional PLIE/path": row.get("e_directional_pressure", 0),
                "jump/volatility instability": max(row.get("e_jump", 0), row.get("e_volatility", 0)),
                "1h shock aligned with PLIE": row.get("e_shock_aligned_plie", 0),
                "strong entry/accel": max(row.get("e_strong_entry", 0), row.get("e_accel", 0)),
            },
            "RHA": {
                "strict pressure rejection": row.get("e_rha_strict", 0),
                "pressure rejection": row.get("e_rejection", 0),
                "absorption": row.get("e_absorption", 0),
                "1h shock against PLIE": row.get("e_shock_against_plie", 0),
                "directional pressure still exists": row.get("e_directional_pressure", 0),
                "low cascade": 1 - row.get("e_cascade", 0),
            },
            "VT": {
                "active dominance": row.get("e_active", 0),
                "1h active shock breakout": row.get("e_shock_active_breakout", 0),
                "trend strength": row.get("e_trend_strength", 0),
                "volatility instability": row.get("e_volatility", 0),
                "neutral/mixed pressure": row.get("e_neutral_or_mixed", 0),
                "low directional PLIE explanation": 1 - row.get("e_directional_pressure", 0),
            },
            "RC": {
                "quiet no-pressure": row.get("e_quiet", 0),
                "neutral pressure": row.get("e_neutral_context", 0),
                "low realized volatility": row.get("e_low_vol", 0),
                "range compression": row.get("e_range_compression", 0),
                "low trend": row.get("e_low_trend", 0),
            },
            "AMB": {
                "evidence conflict": row.get("e_conflict", 0),
                "data-quality/staleness risk": row.get("e_quality_bad", 0),
                "cross-window conflict": row.get("e_cross_window_conflict", 0),
                "mixed pressure/chop": max(row.get("e_mixed_context", 0), row.get("e_label_margin_low", 0)),
                "weak semantic clarity": 1 - max(row.get("e_baseline", 0), row.get("e_cascade", 0), row.get("e_rejection", 0), row.get("e_active", 0), row.get("e_quiet", 0)),
            },
        }
        vals = maps.get(state, {})
        ranked = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return "; ".join([f"{k}={float(v):.3f}" for k, v in ranked])
