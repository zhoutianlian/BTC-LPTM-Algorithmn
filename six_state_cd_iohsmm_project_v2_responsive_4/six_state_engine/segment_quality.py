from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from .constants import STATES


def _num(s: pd.Series | Any, default: float = 0.0) -> pd.Series | float:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    try:
        x = float(s)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _clip01(x):
    if isinstance(x, pd.Series):
        return x.clip(0.0, 1.0)
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_col(df: pd.DataFrame, col: str, default: float | str | pd.Series = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col]
    # Important: defaults are sometimes fallback Series, e.g. trend_strength_6h -> e_trend_strength.
    # Do not create a Series-of-Series, which is extremely slow and memory-heavy.
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)


def _indicator(cond: pd.Series) -> pd.Series:
    return cond.fillna(False).astype(float)


def _scale_to_threshold(s: pd.Series, threshold: float) -> pd.Series:
    threshold = max(float(threshold), 1e-9)
    return _clip01(_num(s, 0.0) / threshold)


def _movement_score(abs_ret: pd.Series, range_bps: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    min_abs = float(cfg.get("min_abs_return_bps", 30.0))
    min_range = float(cfg.get("min_range_bps", 60.0))
    return pd.concat([
        _scale_to_threshold(abs_ret, min_abs),
        _scale_to_threshold(range_bps, min_range),
    ], axis=1).max(axis=1)


def _price_impact_score(df: pd.DataFrame, abs_ret: pd.Series, range_bps: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    rv_thr = float(cfg.get("min_realized_vol_1h_bps", 45.0))
    jump_thr = float(cfg.get("min_jump_proxy_1h", 0.30))
    min_abs = float(cfg.get("min_abs_return_bps", 30.0))
    min_range = float(cfg.get("min_range_bps", 60.0))
    parts = [
        _scale_to_threshold(abs_ret, min_abs),
        _scale_to_threshold(range_bps, min_range),
        _scale_to_threshold(_safe_col(df, "realized_vol_1h_bps", 0.0), rv_thr),
        _scale_to_threshold(_safe_col(df, "jump_proxy_1h", 0.0), jump_thr),
        _num(_safe_col(df, "e_short_impulse_1h", 0.0), 0.0),
    ]
    return pd.concat(parts, axis=1).max(axis=1).clip(0.0, 1.0)


def apply_state_confirmation(df: pd.DataFrame, model_config: dict[str, Any]) -> pd.DataFrame:
    """Add causal per-row state confirmation and trading-trigger quality fields.

    This layer intentionally does not overwrite `stable_state`. It uses only information
    available from the current stable-state entry up to the current row, so it is suitable
    for live inspection and trading-trigger gating.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    cfg = model_config.get("state_confirmation", {}) if isinstance(model_config.get("state_confirmation", {}), dict) else {}
    enabled = bool(cfg.get("enabled", True))
    default_cfg = cfg.get("defaults", {}) if isinstance(cfg.get("defaults", {}), dict) else {}
    state_cfgs = cfg.get("states", {}) if isinstance(cfg.get("states", {}), dict) else {}

    # Defaults are deliberately conservative, but fully configurable.
    default_confirm_bars = int(default_cfg.get("confirm_bars", 3))
    default_min_quality = float(default_cfg.get("min_quality_score", 0.60))
    default_weak_quality = float(default_cfg.get("weak_quality_score", 0.45))

    if "stable_state" not in out.columns:
        raise ValueError("apply_state_confirmation requires stable_state column")

    change = out["stable_state"].astype(str).ne(out["stable_state"].astype(str).shift()).cumsum()
    out["segment_id"] = change.astype(int)
    out["bars_since_state_entry"] = out.groupby("segment_id").cumcount() + 1
    out["state_entry_time"] = out.groupby("segment_id")["time"].transform("first")
    out["state_entry_price"] = out.groupby("segment_id")["price"].transform("first")

    # Causal price path since state entry.
    out["entry_to_current_return_bps"] = 10000.0 * np.log(out["price"] / out["state_entry_price"])
    out["entry_to_current_abs_return_bps"] = out["entry_to_current_return_bps"].abs()
    running_high = out.groupby("segment_id")["price"].cummax()
    running_low = out.groupby("segment_id")["price"].cummin()
    out["entry_to_current_high"] = running_high
    out["entry_to_current_low"] = running_low
    out["entry_to_current_range_bps"] = 10000.0 * np.log(running_high / running_low)
    # MFE/MAE are not assumed trade signals; they describe path excursion since state entry.
    out["entry_to_current_mfe_bps"] = 10000.0 * np.log(running_high / out["state_entry_price"])
    out["entry_to_current_mae_bps"] = 10000.0 * np.log(running_low / out["state_entry_price"])

    state = out["stable_state"].astype(str)
    abs_ret = _num(out["entry_to_current_abs_return_bps"], 0.0)
    range_bps = _num(out["entry_to_current_range_bps"], 0.0)
    bars = _num(out["bars_since_state_entry"], 1.0)

    # Base semantic evidence.
    e_neutral = _num(_safe_col(out, "e_neutral_context", 0.0), 0.0)
    e_quiet = _num(_safe_col(out, "e_quiet", 0.0), 0.0)
    e_low_vol = _num(_safe_col(out, "e_low_vol", 0.0), 0.0)
    e_range_comp = _num(_safe_col(out, "e_range_compression", 0.0), 0.0)
    e_low_trend = _num(_safe_col(out, "e_low_trend", 0.0), 0.0)
    e_low_jump = _num(_safe_col(out, "e_low_jump", 0.0), 0.0)
    e_conflict = _num(_safe_col(out, "e_conflict", 0.0), 0.0)
    e_quality_bad = _num(_safe_col(out, "e_quality_bad", 0.0), 0.0)
    e_active = _num(_safe_col(out, "e_active", 0.0), 0.0)
    e_directional = _num(_safe_col(out, "e_directional_pressure", 0.0), 0.0)
    e_cascade = _num(_safe_col(out, "e_cascade", 0.0), 0.0)
    e_pressure_strength = _num(_safe_col(out, "e_pressure_strength", 0.0), 0.0)
    e_rha_strict = _num(_safe_col(out, "e_rha_strict", 0.0), 0.0)
    e_baseline = _num(_safe_col(out, "e_baseline", 0.0), 0.0)
    e_orderly = _num(_safe_col(out, "e_orderly_trend", 0.0), 0.0)
    e_volatility = _num(_safe_col(out, "e_volatility", 0.0), 0.0)
    e_trend_strength = _num(_safe_col(out, "e_trend_strength", 0.0), 0.0)
    e_trend_cons = _num(_safe_col(out, "e_trend_consistency", 0.0), 0.0)
    e_neutral_mixed = _num(_safe_col(out, "e_neutral_or_mixed", 0.0), 0.0)
    e_absorption = _num(_safe_col(out, "e_absorption", 0.0), 0.0)
    e_rejection = _num(_safe_col(out, "e_rejection", 0.0), 0.0)

    trend_strength_6h = _num(_safe_col(out, "trend_strength_6h", out.get("e_trend_strength", 0.0)), 0.0)
    trend_cons_6h = _num(_safe_col(out, "trend_consistency_6h", out.get("e_trend_consistency", 0.0)), 0.0)
    range_comp_6h = _num(_safe_col(out, "range_compression_6h", out.get("e_range_compression", 0.0)), 0.0)
    jump_proxy_1h = _num(_safe_col(out, "jump_proxy_1h", 0.0), 0.0)

    score = pd.Series(0.5, index=out.index, dtype=float)
    reason = pd.Series("generic", index=out.index, dtype=object)

    # RC: low pressure + quiet + compression + low trend; strong trends exclude high-quality RC.
    rc_cfg = state_cfgs.get("RC", {}) if isinstance(state_cfgs.get("RC", {}), dict) else {}
    rc_trend_exclusion = (
        ((trend_strength_6h > float(rc_cfg.get("max_trend_strength_6h", 0.50))) & (trend_cons_6h > float(rc_cfg.get("max_trend_consistency_6h", 0.58))))
        | (e_active > float(rc_cfg.get("max_active_evidence", 0.55)))
        | (range_comp_6h < float(rc_cfg.get("min_range_compression_6h", 0.45)))
    )
    rc_score = (
        0.20 * e_neutral + 0.20 * e_quiet + 0.16 * e_low_vol + 0.16 * e_range_comp
        + 0.13 * e_low_trend + 0.08 * e_low_jump + 0.07 * (1 - e_conflict)
        - 0.25 * rc_trend_exclusion.astype(float) - 0.10 * e_quality_bad
    ).clip(0, 1)
    m = state.eq("RC")
    score.loc[m] = rc_score.loc[m]
    reason.loc[m] = "RC quality: neutral/quiet + low vol + compression + low trend; trend exclusion penalizes"

    # HPEM: needs pressure/cascade AND price impact.
    hpem_cfg = state_cfgs.get("HPEM", {}) if isinstance(state_cfgs.get("HPEM", {}), dict) else {}
    hpem_price_impact = _price_impact_score(out, abs_ret, range_bps, hpem_cfg)
    hpem_score = (
        0.24 * e_directional + 0.24 * e_cascade + 0.18 * e_pressure_strength
        + 0.24 * hpem_price_impact + 0.10 * (1 - e_rha_strict)
        - 0.10 * e_conflict
    ).clip(0, 1)
    m = state.eq("HPEM")
    score.loc[m] = hpem_score.loc[m]
    reason.loc[m] = "HPEM quality: directional pressure + cascade + price impact/vol/jump confirmation"
    out["hpem_price_impact_score"] = hpem_price_impact

    # VT: active trend/breakout plus movement/range expansion, not just volatility.
    vt_cfg = state_cfgs.get("VT", {}) if isinstance(state_cfgs.get("VT", {}), dict) else {}
    vt_movement = pd.concat([
        _movement_score(abs_ret, range_bps, vt_cfg),
        _num(trend_strength_6h, 0.0),
        _num(trend_cons_6h, 0.0),
        (1 - _num(range_comp_6h, 0.0)).clip(0, 1),
    ], axis=1).mean(axis=1).clip(0, 1)
    vt_score = (
        0.30 * e_active + 0.18 * e_neutral_mixed + 0.22 * e_trend_strength
        + 0.20 * vt_movement + 0.10 * e_volatility
        - 0.12 * (e_directional * e_cascade) - 0.08 * e_conflict
    ).clip(0, 1)
    m = state.eq("VT")
    score.loc[m] = vt_score.loc[m]
    reason.loc[m] = "VT quality: active dominance + trend/vol + causal movement/range expansion"
    out["vt_movement_confirmation_score"] = vt_movement

    # ST: orderly trend requires trend confirmation; avoid weak fallback.
    st_cfg = state_cfgs.get("ST", {}) if isinstance(state_cfgs.get("ST", {}), dict) else {}
    st_movement = _movement_score(abs_ret, range_bps, st_cfg)
    st_score = (
        0.22 * e_baseline + 0.22 * e_orderly + 0.20 * e_trend_strength
        + 0.15 * e_trend_cons + 0.14 * st_movement + 0.07 * (1 - e_conflict)
        - 0.10 * e_quiet - 0.10 * e_rha_strict - 0.10 * e_cascade
    ).clip(0, 1)
    m = state.eq("ST")
    score.loc[m] = st_score.loc[m]
    reason.loc[m] = "ST quality: baseline/orderly trend + trend strength/consistency + causal movement"
    out["st_movement_confirmation_score"] = st_movement

    # RHA and AMB are less central to this request, but keep quality diagnostics coherent.
    rha_score = (0.28 * e_directional + 0.30 * e_rha_strict + 0.18 * e_rejection + 0.14 * e_absorption + 0.10 * (1 - e_cascade)).clip(0, 1)
    m = state.eq("RHA")
    score.loc[m] = rha_score.loc[m]
    reason.loc[m] = "RHA quality: directional pressure + rejection/takeover/absorption"

    amb_score = pd.concat([e_conflict, e_quality_bad, _num(_safe_col(out, "e_cross_window_conflict", 0.0), 0.0)], axis=1).max(axis=1).clip(0, 1)
    m = state.eq("AMB")
    score.loc[m] = amb_score.loc[m]
    reason.loc[m] = "AMB quality: conflict/data-quality/cross-window uncertainty"

    if not enabled:
        score[:] = 1.0
        reason[:] = "state_confirmation disabled"

    out["state_confirmation_score"] = score.clip(0, 1)
    out["state_quality_reason"] = reason

    # State-specific thresholds.
    confirm_bars = pd.Series(default_confirm_bars, index=out.index, dtype=float)
    min_q = pd.Series(default_min_quality, index=out.index, dtype=float)
    weak_q = pd.Series(default_weak_quality, index=out.index, dtype=float)
    for s in STATES:
        scfg = state_cfgs.get(s, {}) if isinstance(state_cfgs.get(s, {}), dict) else {}
        mask = state.eq(s)
        confirm_bars.loc[mask] = float(scfg.get("confirm_bars", default_confirm_bars))
        min_q.loc[mask] = float(scfg.get("min_quality_score", default_min_quality))
        weak_q.loc[mask] = float(scfg.get("weak_quality_score", default_weak_quality))

    # Immediate confirmation for high-quality HPEM shock/cascade and high-quality RC sustained quiet.
    immediate = pd.Series(False, index=out.index)
    hcfg = state_cfgs.get("HPEM", {}) if isinstance(state_cfgs.get("HPEM", {}), dict) else {}
    immediate |= state.eq("HPEM") & (hpem_price_impact >= float(hcfg.get("immediate_min_price_impact", 0.70))) & (e_cascade >= float(hcfg.get("immediate_min_cascade", 0.50)))

    pending = (bars < confirm_bars) & ~immediate
    confirmed = ((score >= min_q) & (~pending)) | immediate
    weak = (score < weak_q) | ((state.eq("RC")) & rc_trend_exclusion) | ((state.eq("HPEM")) & (hpem_price_impact < float(hpem_cfg.get("min_price_impact", 0.45))))
    # Pending high-quality states are not weak, but should not be full trading triggers yet.
    weak = weak & ~pending

    out["state_confirm_bars_required"] = confirm_bars.astype(int)
    out["state_min_quality_required"] = min_q
    out["state_pending_confirmation_flag"] = pending.astype(int)
    out["state_confirmed_flag"] = confirmed.astype(int)
    out["weak_state_flag"] = weak.astype(int)

    status = pd.Series("confirmed", index=out.index, dtype=object)
    status[pending] = "pending"
    status[weak] = "weak"
    status[(~confirmed) & (~pending) & (~weak)] = "unconfirmed"
    out["state_confirmation_status"] = status

    # Trading trigger quality should be conservative for pending/weak states but preserve information.
    q = score.copy()
    q[pending] *= float(default_cfg.get("pending_quality_multiplier", 0.55))
    q[weak] *= float(default_cfg.get("weak_quality_multiplier", 0.35))
    q[status.eq("unconfirmed")] *= float(default_cfg.get("unconfirmed_quality_multiplier", 0.70))
    out["trading_trigger_quality"] = q.clip(0, 1)
    out["trading_trigger_state"] = np.where(confirmed, out["stable_state"], "WEAK_" + out["stable_state"].astype(str))
    out.loc[pending, "trading_trigger_state"] = "PENDING_" + out.loc[pending, "stable_state"].astype(str)

    # Human-readable issue flag for row-level dashboard and downstream audit.
    issue = pd.Series("", index=out.index, dtype=object)
    issue[state.eq("RC") & rc_trend_exclusion] = "RC trend exclusion"
    issue[state.eq("HPEM") & (hpem_price_impact < float(hpem_cfg.get("min_price_impact", 0.45)))] = "HPEM weak price impact"
    issue[state.eq("VT") & (vt_movement < float(vt_cfg.get("min_movement_confirmation", 0.45)))] = "VT weak movement confirmation"
    issue[state.eq("ST") & (st_score < float(st_cfg.get("weak_quality_score", 0.45)))] = "ST weak trend confirmation"
    out["state_quality_issue"] = issue
    return out


SEGMENT_NUMERIC_COLS = [
    "realized_vol_1h_bps", "realized_vol_6h_bps", "trend_strength_1h", "trend_strength_6h",
    "trend_consistency_1h", "trend_consistency_6h", "range_compression_6h",
    "jump_proxy_1h", "jump_proxy_6h", "state_confirmation_score", "trading_trigger_quality",
    "directional_trading_quality", "direction_confidence",
    "hpem_price_impact_score", "vt_movement_confirmation_score", "st_movement_confirmation_score",
    "hpem_directional_compliance_score", "hpem_aligned_move_bps", "hpem_aligned_mfe_bps",
    "hpem_giveback_bps", "hpem_giveback_ratio",
    "rha_directional_compliance_score", "rha_reversal_move_bps", "rha_reversal_mfe_bps",
    "rha_follow_plie_move_bps", "rha_reversal_score",
    "vt_direction_score", "vt_direction_confidence", "vt_up_ratio_recent", "vt_down_ratio_recent",
    "state_price_expectation_score", "price_expectation_exit_pressure", "price_expectation_trading_quality",
    "price_expectation_raw_issue_flag", "price_expectation_exit_flag", "execution_quality",
    "rc_price_drift_flag", "rc_range_break_flag", "rc_trend_exclusion_flag",
    "vt_low_move_flag", "vt_mixed_direction_flag", "vt_weak_trend_flag",
    "st_flat_flag", "st_weak_trend_flag", "st_too_extreme_flag",
    "hpem_price_expectation_fail_flag", "rha_price_expectation_fail_flag",
]


def _mode_value(s: pd.Series) -> Any:
    if s.empty:
        return np.nan
    m = s.dropna().astype(str).mode()
    return m.iloc[0] if not m.empty else np.nan


def compute_segment_diagnostics(df: pd.DataFrame, model_config: dict[str, Any] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    if "segment_id" not in d.columns:
        d["segment_id"] = d["stable_state"].astype(str).ne(d["stable_state"].astype(str).shift()).cumsum().astype(int)
    rows: list[dict[str, Any]] = []
    for seg_id, g in d.groupby("segment_id", sort=True):
        g = g.sort_values("time")
        state = str(g["stable_state"].iloc[0])
        p = pd.to_numeric(g["price"], errors="coerce")
        entry = float(p.iloc[0]) if len(p) else np.nan
        exitp = float(p.iloc[-1]) if len(p) else np.nan
        high = float(p.max()) if p.notna().any() else np.nan
        low = float(p.min()) if p.notna().any() else np.nan
        ret = float(10000.0 * np.log(exitp / entry)) if entry and exitp and np.isfinite(entry) and np.isfinite(exitp) and entry > 0 and exitp > 0 else np.nan
        rng = float(10000.0 * np.log(high / low)) if high and low and np.isfinite(high) and np.isfinite(low) and high > 0 and low > 0 else np.nan
        if np.isfinite(ret) and ret >= 0:
            mfe = float(10000.0 * np.log(high / entry)) if entry > 0 and high > 0 else np.nan
            mae = float(10000.0 * np.log(low / entry)) if entry > 0 and low > 0 else np.nan
        elif np.isfinite(ret):
            mfe = float(10000.0 * np.log(entry / low)) if entry > 0 and low > 0 else np.nan
            mae = float(10000.0 * np.log(entry / high)) if entry > 0 and high > 0 else np.nan
        else:
            mfe = mae = np.nan
        start = g["time"].iloc[0]
        end = g["time"].iloc[-1]
        duration_min = float((end - start).total_seconds() / 60.0) if pd.notna(start) and pd.notna(end) else np.nan
        row: dict[str, Any] = {
            "segment_id": int(seg_id),
            "state": state,
            "start_time": start,
            "end_time": end,
            "duration_min": duration_min,
            "duration_bar_count": int(len(g)),
            "entry_price": entry,
            "exit_price": exitp,
            "segment_return_bps": ret,
            "abs_segment_return_bps": abs(ret) if np.isfinite(ret) else np.nan,
            "segment_high": high,
            "segment_low": low,
            "segment_range_bps": rng,
            "max_favorable_move_bps": mfe,
            "max_adverse_move_bps": mae,
            "dominant_path_context_6h": _mode_value(g.get("path_context_6h", pd.Series(dtype=object))),
            "dominant_path_label_6h": _mode_value(g.get("path_label_6h", pd.Series(dtype=object))),
            "dominant_path_context_24h": _mode_value(g.get("path_context_24h", pd.Series(dtype=object))),
            "dominant_path_label_24h": _mode_value(g.get("path_label_24h", pd.Series(dtype=object))),
            "hmm_state_mode": _mode_value(g.get("hmm_state", pd.Series(dtype=object))),
            "switch_trigger_type": _mode_value(g.get("from_state", pd.Series(dtype=object))) if len(g) else "",
            "switch_reason": str(g.get("transition_reason", pd.Series([""])).iloc[0]) if "transition_reason" in g else "",
            "dominant_directional_state": _mode_value(g.get("directional_state", pd.Series(dtype=object))),
            "dominant_trade_direction": _mode_value(g.get("trade_direction", pd.Series(dtype=object))),
            "dominant_execution_state": _mode_value(g.get("execution_state", pd.Series(dtype=object))),
            "dominant_execution_trade_direction": _mode_value(g.get("execution_trade_direction", pd.Series(dtype=object))),
            "dominant_price_expectation_issue": _mode_value(g.get("state_price_expectation_issue", pd.Series(dtype=object))),
            "dominant_price_expectation_exit_target": _mode_value(g.get("price_expectation_exit_target", pd.Series(dtype=object))),
            "directional_exit_ratio": float(pd.to_numeric(g["directional_exit_trigger_flag"], errors="coerce").fillna(0).mean()) if "directional_exit_trigger_flag" in g else 0.0,
            "price_expectation_exit_ratio": float(pd.to_numeric(g["price_expectation_exit_flag"], errors="coerce").fillna(0).mean()) if "price_expectation_exit_flag" in g else 0.0,
            "confirmed_ratio": float(pd.to_numeric(g["state_confirmed_flag"], errors="coerce").fillna(0).mean()) if "state_confirmed_flag" in g else 0.0,
            "weak_ratio": float(pd.to_numeric(g["weak_state_flag"], errors="coerce").fillna(0).mean()) if "weak_state_flag" in g else 0.0,
            "pending_ratio": float(pd.to_numeric(g["state_pending_confirmation_flag"], errors="coerce").fillna(0).mean()) if "state_pending_confirmation_flag" in g else 0.0,
            "mean_trading_trigger_quality": float(pd.to_numeric(g["trading_trigger_quality"], errors="coerce").mean()) if "trading_trigger_quality" in g else np.nan,
            "min_trading_trigger_quality": float(pd.to_numeric(g["trading_trigger_quality"], errors="coerce").min()) if "trading_trigger_quality" in g else np.nan,
            "mean_directional_trading_quality": float(pd.to_numeric(g["directional_trading_quality"], errors="coerce").mean()) if "directional_trading_quality" in g else np.nan,
            "min_directional_trading_quality": float(pd.to_numeric(g["directional_trading_quality"], errors="coerce").min()) if "directional_trading_quality" in g else np.nan,
        }
        for c in SEGMENT_NUMERIC_COLS:
            if c in g.columns:
                row[f"mean_{c}"] = float(pd.to_numeric(g[c], errors="coerce").mean())
                row[f"max_{c}"] = float(pd.to_numeric(g[c], errors="coerce").max())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return add_segment_quality_flags(out, model_config or {})


def add_segment_quality_flags(seg: pd.DataFrame, model_config: dict[str, Any]) -> pd.DataFrame:
    out = seg.copy()
    qcfg = model_config.get("segment_quality", {}) if isinstance(model_config.get("segment_quality", {}), dict) else {}
    short_bars = int(qcfg.get("short_segment_bars", 6))
    low_move_bps = float(qcfg.get("low_move_bps", 30.0))
    low_range_bps = float(qcfg.get("low_range_bps", 60.0))
    rc_trend_strength = float(qcfg.get("rc_high_trend_strength_6h", 0.50))
    rc_trend_cons = float(qcfg.get("rc_high_trend_consistency_6h", 0.58))
    weak_trend_strength = float(qcfg.get("weak_trend_strength_6h", 0.38))
    weak_trend_cons = float(qcfg.get("weak_trend_consistency_6h", 0.50))
    hpem_min_impact = float(qcfg.get("hpem_min_price_impact", 0.45))
    vt_min_move_score = float(qcfg.get("vt_min_movement_confirmation", 0.45))

    n = lambda c: pd.to_numeric(out.get(c, np.nan), errors="coerce")
    out["q_short_low_move"] = ((n("duration_bar_count") <= short_bars) & (n("abs_segment_return_bps") < low_move_bps) & (n("segment_range_bps") < low_range_bps)).astype(int)
    out["q_rc_trending"] = ((out["state"].eq("RC")) & (n("mean_trend_strength_6h") > rc_trend_strength) & (n("mean_trend_consistency_6h") > rc_trend_cons)).astype(int)
    out["q_hpem_weak_price_impact"] = ((out["state"].eq("HPEM")) & (n("mean_hpem_price_impact_score") < hpem_min_impact)).astype(int)
    out["q_hpem_directional_decay"] = ((out["state"].eq("HPEM")) & ((n("max_hpem_giveback_bps") > float(qcfg.get("hpem_giveback_bps", 120.0))) | (n("max_hpem_aligned_mfe_bps") < float(qcfg.get("hpem_min_aligned_mfe_bps", 80.0))))).astype(int)
    out["q_rha_invalid_direction"] = ((out["state"].eq("RHA")) & (n("directional_exit_ratio") > 0.05)).astype(int)
    out["q_vt_weak_movement"] = ((out["state"].eq("VT")) & (n("mean_vt_movement_confirmation_score") < vt_min_move_score)).astype(int)
    out["q_vt_weak_trend"] = ((out["state"].eq("VT")) & ((n("mean_trend_strength_6h") < weak_trend_strength) | (n("mean_trend_consistency_6h") < weak_trend_cons))).astype(int)
    out["q_vt_mixed_direction"] = ((out["state"].eq("VT")) & (out.get("dominant_directional_state", "").astype(str).eq("VT_MIXED"))).astype(int)
    out["q_st_weak_trend"] = ((out["state"].eq("ST")) & ((n("mean_trend_strength_6h") < weak_trend_strength) | (n("mean_trend_consistency_6h") < weak_trend_cons))).astype(int)
    out["q_low_trading_quality"] = (n("mean_trading_trigger_quality") < float(qcfg.get("low_trading_quality", 0.45))).astype(int)
    out["q_low_directional_quality"] = (n("mean_directional_trading_quality") < float(qcfg.get("low_directional_quality", 0.45))).astype(int)
    out["q_low_price_expectation_quality"] = (n("mean_price_expectation_trading_quality") < float(qcfg.get("low_price_expectation_quality", 0.45))).astype(int)
    out["q_price_expectation_exit"] = (n("mean_price_expectation_exit_flag") > float(qcfg.get("price_expectation_exit_ratio", 0.05))).astype(int)
    out["q_rc_price_too_large"] = ((out["state"].eq("RC")) & ((n("mean_rc_price_drift_flag") > 0.05) | (n("mean_rc_range_break_flag") > 0.05) | (n("mean_rc_trend_exclusion_flag") > 0.05))).astype(int)
    out["q_vt_low_move_price"] = ((out["state"].eq("VT")) & ((n("mean_vt_low_move_flag") > 0.05) | (n("mean_vt_weak_trend_flag") > 0.05))).astype(int)
    out["q_st_price_expectation"] = ((out["state"].eq("ST")) & ((n("mean_st_flat_flag") > 0.05) | (n("mean_st_weak_trend_flag") > 0.05) | (n("mean_st_too_extreme_flag") > 0.05))).astype(int)
    out["q_hpem_price_expectation_exit"] = ((out["state"].eq("HPEM")) & ((n("mean_hpem_price_no_response_exit_flag") > 0.05) | (n("mean_hpem_price_decay_exit_flag") > 0.05) | (n("mean_hpem_price_adverse_exit_flag") > 0.05))).astype(int)
    out["q_rha_price_expectation_exit"] = ((out["state"].eq("RHA")) & ((n("mean_rha_invalid_follow_plie_exit_flag") > 0.05) | (n("mean_rha_weak_no_reversal_flag") > 0.05))).astype(int)

    # V2.5 price-expectation audit flags.
    out["q_price_expectation_exit"] = (n("price_expectation_exit_ratio") > float(qcfg.get("price_expectation_exit_ratio", 0.05))).astype(int)
    out["q_price_expectation_low_score"] = (n("mean_state_price_expectation_score") < float(qcfg.get("low_price_expectation_score", 0.45))).astype(int)
    out["q_hpem_px_failure"] = ((out["state"].eq("HPEM")) & (n("max_hpem_price_expectation_fail_flag") > 0)).astype(int)
    out["q_rha_px_failure"] = ((out["state"].eq("RHA")) & (n("max_rha_price_expectation_fail_flag") > 0)).astype(int)
    out["q_rc_price_too_large"] = ((out["state"].eq("RC")) & ((n("max_rc_price_drift_flag") > 0) | (n("max_rc_range_break_flag") > 0))).astype(int)
    out["q_rc_price_not_small"] = ((out["state"].eq("RC")) & ((n("abs_segment_return_bps") > float(qcfg.get("rc_max_abs_return_bps", 150.0))) | (n("segment_range_bps") > float(qcfg.get("rc_max_range_bps", 260.0))))).astype(int)
    out["q_vt_low_move_px"] = ((out["state"].eq("VT")) & ((n("max_vt_low_move_flag") > 0) | ((n("abs_segment_return_bps") < float(qcfg.get("vt_min_abs_return_bps", 60.0))) & (n("segment_range_bps") < float(qcfg.get("vt_min_range_bps", 120.0)))))).astype(int)
    out["q_vt_mixed_or_weak"] = ((out["state"].eq("VT")) & ((n("max_vt_mixed_direction_flag") > 0) | (n("max_vt_weak_trend_flag") > 0))).astype(int)
    out["q_st_price_expectation_fail"] = ((out["state"].eq("ST")) & ((n("max_st_flat_flag") > 0) | (n("max_st_weak_trend_flag") > 0) | (n("max_st_too_extreme_flag") > 0))).astype(int)
    out["q_st_flat_or_extreme"] = ((out["state"].eq("ST")) & ((((n("abs_segment_return_bps") < float(qcfg.get("st_min_abs_return_bps", 35.0))) & (n("segment_range_bps") < float(qcfg.get("st_min_range_bps", 70.0)))) | (n("abs_segment_return_bps") > float(qcfg.get("st_max_abs_return_bps", 220.0))) | (n("segment_range_bps") > float(qcfg.get("st_max_range_bps", 420.0)))))).astype(int)

    issue_cols = [c for c in out.columns if c.startswith("q_")]
    out["segment_quality_issue_count"] = out[issue_cols].sum(axis=1)
    labels = []
    for _, r in out.iterrows():
        labs = [c.replace("q_", "") for c in issue_cols if int(r.get(c, 0)) == 1]
        labels.append("; ".join(labs))
    out["segment_quality_issues"] = labels
    return out


def segment_quality_summary(seg: pd.DataFrame) -> dict[str, Any]:
    if seg.empty:
        return {}
    issue_cols = [c for c in seg.columns if c.startswith("q_")]
    by_state: dict[str, Any] = {}
    for state, g in seg.groupby("state"):
        by_state[state] = {
            "segments": int(len(g)),
            "median_abs_return_bps": float(pd.to_numeric(g.get("abs_segment_return_bps"), errors="coerce").median()),
            "median_range_bps": float(pd.to_numeric(g.get("segment_range_bps"), errors="coerce").median()),
            "mean_trading_trigger_quality": float(pd.to_numeric(g.get("mean_trading_trigger_quality"), errors="coerce").mean()),
            "mean_directional_trading_quality": float(pd.to_numeric(g.get("mean_directional_trading_quality"), errors="coerce").mean()),
            "mean_price_expectation_score": float(pd.to_numeric(g.get("mean_state_price_expectation_score"), errors="coerce").mean()),
            "price_expectation_exit_ratio": float(pd.to_numeric(g["price_expectation_exit_ratio"], errors="coerce").fillna(0).mean()) if "price_expectation_exit_ratio" in g else 0.0,
            "directional_exit_ratio": float(pd.to_numeric(g["directional_exit_ratio"], errors="coerce").fillna(0).mean()) if "directional_exit_ratio" in g else 0.0,
            "weak_segment_ratio": float((pd.to_numeric(g["weak_ratio"], errors="coerce").fillna(0) > 0.5).mean()) if "weak_ratio" in g else 0.0,
            "issue_counts": {c: int(pd.to_numeric(g[c], errors="coerce").fillna(0).sum()) for c in issue_cols if c in g},
        }
    return {
        "segments": int(len(seg)),
        "issue_counts": {c: int(pd.to_numeric(seg[c], errors="coerce").fillna(0).sum()) for c in issue_cols},
        "by_state": by_state,
    }


def save_segment_quality_outputs(output_dir: str | Path, df: pd.DataFrame, model_config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    seg = compute_segment_diagnostics(df, model_config)
    seg.to_csv(output_dir / "segment_diagnostics.csv", index=False)
    if not seg.empty:
        flags = seg[seg.get("segment_quality_issue_count", 0) > 0].copy()
        flags.to_csv(output_dir / "segment_quality_flags.csv", index=False)
        examples = flags.sort_values(["segment_quality_issue_count", "duration_bar_count"], ascending=[False, True]).head(200)
        examples.to_csv(output_dir / "low_quality_segment_examples.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "segment_quality_flags.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "low_quality_segment_examples.csv", index=False)
    summary = segment_quality_summary(seg)
    (output_dir / "segment_quality_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary
