from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _safe_series(df: pd.DataFrame, col: str, default: Any = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_safe_series(df, col, default), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _clip01(x: pd.Series | float) -> pd.Series | float:
    return np.clip(x, 0.0, 1.0)


def _scale_down(x: pd.Series, threshold: float) -> pd.Series:
    threshold = max(float(threshold), 1e-9)
    return (1.0 - (x / threshold)).clip(0, 1)


def _scale_up(x: pd.Series, threshold: float) -> pd.Series:
    threshold = max(float(threshold), 1e-9)
    return (x / threshold).clip(0, 1)


def apply_price_expectation_compliance(df: pd.DataFrame, model_config: dict[str, Any]) -> pd.DataFrame:
    """Add causal price-behavior compliance diagnostics and quality penalties.

    This layer uses only state-entry-to-current path information and already-available price
    context. It does not overwrite stable_state. Strong flags can be consumed by the stability
    layer on a fresh pipeline run; post-stability it provides row-level and segment-level audit
    fields for trading quality and dashboard review.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    cfg = model_config.get("state_price_expectation", {}) if isinstance(model_config.get("state_price_expectation", {}), dict) else {}
    enabled = bool(cfg.get("enabled", True))
    states_cfg = cfg.get("states", {}) if isinstance(cfg.get("states", {}), dict) else {}

    if "stable_state" not in out.columns:
        return out
    out["price"] = pd.to_numeric(_safe_series(out, "price", np.nan), errors="coerce")
    if "segment_id" not in out.columns:
        out["segment_id"] = out["stable_state"].astype(str).ne(out["stable_state"].astype(str).shift()).cumsum().astype(int)
    seg = out["segment_id"]
    if "bars_since_state_entry" not in out.columns:
        out["bars_since_state_entry"] = out.groupby("segment_id", sort=False).cumcount() + 1
    if "state_entry_price" not in out.columns:
        out["state_entry_price"] = out.groupby("segment_id", sort=False)["price"].transform("first")
    if "entry_to_current_return_bps" not in out.columns:
        out["entry_to_current_return_bps"] = 10000.0 * np.log(out["price"] / pd.to_numeric(out["state_entry_price"], errors="coerce"))
    if "entry_to_current_range_bps" not in out.columns:
        high = out.groupby("segment_id", sort=False)["price"].cummax()
        low = out.groupby("segment_id", sort=False)["price"].cummin()
        out["entry_to_current_range_bps"] = 10000.0 * np.log(high / low)

    stable = out["stable_state"].astype(str)
    abs_ret = _num(out, "entry_to_current_abs_return_bps", 0.0)
    if "entry_to_current_abs_return_bps" not in out.columns:
        abs_ret = _num(out, "entry_to_current_return_bps", 0.0).abs()
        out["entry_to_current_abs_return_bps"] = abs_ret
    rng = _num(out, "entry_to_current_range_bps", 0.0)
    bars = _num(out, "bars_since_state_entry", 1.0)
    base_q = _num(out, "trading_trigger_quality", 0.5).clip(0, 1)

    # Context fields.
    trend6 = _num(out, "trend_strength_6h", _num(out, "e_trend_strength", 0.0)).clip(0, 1)
    cons6 = _num(out, "trend_consistency_6h", _num(out, "e_trend_consistency", 0.0)).clip(0, 1)
    range_comp6 = _num(out, "range_compression_6h", _num(out, "e_range_compression", 0.0)).clip(0, 1)
    rv1h = _num(out, "realized_vol_1h_bps", 0.0)
    rv6h = _num(out, "realized_vol_6h_bps", 0.0)
    e_active = _num(out, "e_active", 0.0).clip(0, 1)
    e_quiet = _num(out, "e_quiet", 0.0).clip(0, 1)
    e_orderly = _num(out, "e_orderly_trend", 0.0).clip(0, 1)
    e_cascade = _num(out, "e_cascade", 0.0).clip(0, 1)
    e_rha = _num(out, "e_rha_strict", 0.0).clip(0, 1)
    e_conflict = _num(out, "e_conflict", 0.0).clip(0, 1)
    e_abs = _num(out, "e_absorption", 0.0).clip(0, 1)
    e_rej = _num(out, "e_rejection", 0.0).clip(0, 1)
    hpem_impact = _num(out, "hpem_price_impact_score", 0.0).clip(0, 1)
    hpem_decay = _num(out, "hpem_decay_flag", 0).fillna(0) if hasattr(_num(out, "hpem_decay_flag", 0), 'fillna') else _num(out, "hpem_decay_flag", 0)
    hpem_no_resp = _num(out, "hpem_no_response_flag", 0)
    hpem_adverse = _num(out, "hpem_adverse_flag", 0)
    rha_invalid = _num(out, "rha_invalid_follow_plie_flag", 0)
    rha_stall = _num(out, "rha_stall_flag", 0)
    rha_rev = _num(out, "rha_reversal_confirmed_flag", 0)
    vt_mixed = out.get("vt_direction_state", pd.Series("", index=out.index)).astype(str).eq("VT_MIXED") if "vt_direction_state" in out.columns else pd.Series(False, index=out.index)

    # Output defaults.
    out["state_price_expectation_score"] = 0.5
    out["state_price_expectation_issue"] = ""
    out["price_expectation_exit_flag"] = 0
    out["price_expectation_exit_target"] = ""
    out["price_expectation_exit_reason"] = ""
    out["price_expectation_trading_quality"] = base_q
    out["rc_price_drift_flag"] = 0
    out["rc_range_break_flag"] = 0
    out["rc_trend_exclusion_flag"] = 0
    out["vt_low_move_flag"] = 0
    out["vt_mixed_direction_flag"] = 0
    out["st_flat_flag"] = 0
    out["st_weak_trend_flag"] = 0
    out["st_too_extreme_flag"] = 0

    # ---------------- RC: small price changes only ----------------
    rc_cfg = states_cfg.get("RC", {}) if isinstance(states_cfg.get("RC", {}), dict) else {}
    m = stable.eq("RC")
    rc_max_ret = float(rc_cfg.get("max_entry_to_current_abs_return_bps", 80.0))
    rc_max_rng = float(rc_cfg.get("max_entry_to_current_range_bps", 140.0))
    rc_max_tr = float(rc_cfg.get("max_trend_strength_6h", 0.50))
    rc_max_cons = float(rc_cfg.get("max_trend_consistency_6h", 0.58))
    rc_drift = m & (abs_ret > rc_max_ret)
    rc_range = m & (rng > rc_max_rng)
    rc_trend = m & (trend6 > rc_max_tr) & (cons6 > rc_max_cons)
    rc_score = pd.concat([
        _scale_down(abs_ret, rc_max_ret * 1.25),
        _scale_down(rng, rc_max_rng * 1.25),
        (1 - trend6).clip(0, 1),
        (1 - cons6).clip(0, 1),
        range_comp6,
        e_quiet,
    ], axis=1).mean(axis=1).clip(0, 1)
    out.loc[m, "state_price_expectation_score"] = rc_score.loc[m]
    out.loc[rc_drift, "rc_price_drift_flag"] = 1
    out.loc[rc_range, "rc_range_break_flag"] = 1
    out.loc[rc_trend, "rc_trend_exclusion_flag"] = 1
    rc_issue = rc_drift | rc_range | rc_trend
    out.loc[rc_issue, "state_price_expectation_issue"] = "RC_PRICE_TOO_LARGE_OR_TRENDING"
    out.loc[rc_issue, "price_expectation_exit_target"] = np.where(e_active.loc[rc_issue] > 0.40, "VT", np.where(e_orderly.loc[rc_issue] > 0.35, "ST", "VT_OR_ST"))
    out.loc[rc_issue, "price_expectation_exit_reason"] = "RC expected small price/range; current segment drift/range/trend exceeds thresholds"

    # ---------------- HPEM: continuous move in PLIE direction ----------------
    hcfg = states_cfg.get("HPEM", {}) if isinstance(states_cfg.get("HPEM", {}), dict) else {}
    m = stable.eq("HPEM")
    min_impact = float(hcfg.get("min_price_impact", 0.45))
    h_issue = m & ((hpem_no_resp > 0.5) | (hpem_decay > 0.5) | (hpem_adverse > 0.5) | (hpem_impact < min_impact))
    hpem_score = pd.concat([
        hpem_impact,
        _num(out, "hpem_directional_compliance_score", 0.0).clip(0, 1),
        (1 - _num(out, "hpem_giveback_ratio", 0.0).clip(0, 1)),
        _scale_up(rv1h, float(hcfg.get("min_realized_vol_1h_bps", 60.0))),
    ], axis=1).mean(axis=1).clip(0, 1)
    out.loc[m, "state_price_expectation_score"] = hpem_score.loc[m]
    out.loc[h_issue, "state_price_expectation_issue"] = "HPEM_DIRECTION_NOT_CONTINUOUS_OR_NO_IMPACT"
    out.loc[h_issue, "price_expectation_exit_target"] = np.where((e_rej.loc[h_issue] > 0.30) | (e_abs.loc[h_issue] > 0.35) | (e_rha.loc[h_issue] > 0.30), "RHA", np.where(e_active.loc[h_issue] > 0.35, "VT", "RHA_OR_VT"))
    out.loc[h_issue, "price_expectation_exit_reason"] = "HPEM requires continuous PLIE-aligned pressure transmission; no-response/decay/adverse move detected"

    # ---------------- RHA: reversal should be directional for trading; stall is no-trade, follow-PLIE invalidates ----------------
    rcfg = states_cfg.get("RHA", {}) if isinstance(states_cfg.get("RHA", {}), dict) else {}
    m = stable.eq("RHA")
    reversal_min = float(rcfg.get("reversal_min_score", 0.60))
    rha_score = pd.concat([
        _num(out, "rha_directional_compliance_score", 0.0).clip(0, 1),
        _num(out, "rha_reversal_score", 0.0).clip(0, 1),
        e_rej,
        e_abs,
    ], axis=1).mean(axis=1).clip(0, 1)
    rha_issue = m & ((rha_invalid > 0.5) | ((rha_stall > 0.5) & bool(rcfg.get("require_reversal_for_trading", True))) | ((_num(out, "rha_reversal_score", 0.0) < reversal_min) & (rha_rev < 0.5)))
    out.loc[m, "state_price_expectation_score"] = rha_score.loc[m]
    out.loc[m & (rha_stall > 0.5), "state_price_expectation_issue"] = "RHA_STALL_NO_DIRECTIONAL_TRADE"
    out.loc[m & (_num(out, "rha_reversal_score", 0.0) < reversal_min) & (rha_rev < 0.5), "state_price_expectation_issue"] = "RHA_WEAK_NO_REVERSAL"
    out.loc[m & (rha_invalid > 0.5), "state_price_expectation_issue"] = "RHA_INVALID_PRICE_FOLLOWS_PLIE"
    out.loc[rha_issue, "price_expectation_exit_target"] = np.where(rha_invalid.loc[rha_issue] > 0.5, np.where(e_cascade.loc[rha_issue] > 0.35, "HPEM", "ST_OR_HPEM"), np.where(e_quiet.loc[rha_issue] > 0.45, "RC", "RC_OR_AMB"))
    out.loc[rha_issue, "price_expectation_exit_reason"] = "RHA trading expectation requires reversal continuity; stall/weak/price-follow-PLIE detected"

    # ---------------- VT: large move / active range, direction may flip ----------------
    vcfg = states_cfg.get("VT", {}) if isinstance(states_cfg.get("VT", {}), dict) else {}
    m = stable.eq("VT")
    vt_min_ret = float(vcfg.get("min_entry_to_current_abs_return_bps", 60.0))
    vt_min_rng = float(vcfg.get("min_entry_to_current_range_bps", 120.0))
    vt_min_rv1 = float(vcfg.get("min_realized_vol_1h_bps", 45.0))
    vt_min_rv6 = float(vcfg.get("min_realized_vol_6h_bps", 100.0))
    vt_low = m & (abs_ret < vt_min_ret) & (rng < vt_min_rng) & (rv1h < vt_min_rv1) & (rv6h < vt_min_rv6)
    vt_score = pd.concat([
        _scale_up(abs_ret, vt_min_ret),
        _scale_up(rng, vt_min_rng),
        _scale_up(rv1h, vt_min_rv1),
        _scale_up(rv6h, vt_min_rv6),
        trend6,
        e_active,
    ], axis=1).mean(axis=1).clip(0, 1)
    out.loc[m, "state_price_expectation_score"] = vt_score.loc[m]
    out.loc[vt_low, "vt_low_move_flag"] = 1
    out.loc[m & vt_mixed, "vt_mixed_direction_flag"] = 1
    out.loc[vt_low, "state_price_expectation_issue"] = "VT_LOW_MOVE_OR_LOW_VOL"
    out.loc[m & vt_mixed & ~vt_low, "state_price_expectation_issue"] = "VT_MIXED_DIRECTION_NO_DIRECTIONAL_TRADE"
    out.loc[vt_low, "price_expectation_exit_target"] = np.where(e_quiet.loc[vt_low] > 0.40, "RC", "ST_OR_RC")
    out.loc[vt_low, "price_expectation_exit_reason"] = "VT expects large active price movement; current move/range/vol are too small"

    # ---------------- ST: mild orderly trend, not flat and not extreme ----------------
    scfg = states_cfg.get("ST", {}) if isinstance(states_cfg.get("ST", {}), dict) else {}
    m = stable.eq("ST")
    st_min_ret = float(scfg.get("min_entry_to_current_abs_return_bps", 35.0))
    st_min_rng = float(scfg.get("min_entry_to_current_range_bps", 70.0))
    st_min_tr = float(scfg.get("min_trend_strength_6h", 0.38))
    st_min_cons = float(scfg.get("min_trend_consistency_6h", 0.50))
    st_max_ret = float(scfg.get("max_entry_to_current_abs_return_bps", 220.0))
    st_max_rng = float(scfg.get("max_entry_to_current_range_bps", 420.0))
    st_flat = m & (abs_ret < st_min_ret) & (rng < st_min_rng)
    st_weak = m & ((trend6 < st_min_tr) | (cons6 < st_min_cons))
    st_extreme = m & ((abs_ret > st_max_ret) | (rng > st_max_rng) | (e_cascade > 0.55))
    st_score = pd.concat([
        _scale_up(abs_ret, st_min_ret).clip(0, 1),
        _scale_up(rng, st_min_rng).clip(0, 1),
        trend6,
        cons6,
        e_orderly,
        e_cascade.rsub(1.0).clip(0, 1),
    ], axis=1).mean(axis=1).clip(0, 1)
    st_score = st_score * (1 - 0.35 * st_extreme.astype(float))
    out.loc[m, "state_price_expectation_score"] = st_score.loc[m]
    out.loc[st_flat, "st_flat_flag"] = 1
    out.loc[st_weak, "st_weak_trend_flag"] = 1
    out.loc[st_extreme, "st_too_extreme_flag"] = 1
    out.loc[st_flat, "state_price_expectation_issue"] = "ST_FLAT_NO_TREND"
    out.loc[st_weak & ~st_flat, "state_price_expectation_issue"] = "ST_WEAK_TREND"
    out.loc[st_extreme, "state_price_expectation_issue"] = "ST_TOO_EXTREME_FOR_STABLE_TREND"
    st_issue = st_flat | st_weak | st_extreme
    out.loc[st_issue, "price_expectation_exit_target"] = np.where(st_extreme.loc[st_issue], np.where(e_active.loc[st_issue] > 0.35, "VT", "HPEM_OR_VT"), np.where(e_quiet.loc[st_issue] > 0.40, "RC", "RC_OR_AMB"))
    out.loc[st_issue, "price_expectation_exit_reason"] = "ST expects mild orderly trend; flat/weak/extreme behavior detected"

    # Exit flag: post-stability audit. Stability layer uses a causal subset during a fresh run.
    # RHA_STALL is a no-trade condition; it does not always force stable-state exit unless configured.
    exit_raw = pd.Series(False, index=out.index)
    exit_raw |= h_issue
    exit_raw |= (m & False)  # placeholder for readability; m currently ST after section.
    # Rebuild state-specific issue masks robustly.
    exit_raw |= rc_issue & (bars >= int(rc_cfg.get("exit_confirm_bars", 2)))
    exit_raw |= h_issue & (bars >= int(hcfg.get("exit_confirm_bars", 2)))
    exit_raw |= (stable.eq("RHA") & ((rha_invalid > 0.5) | ((rha_stall > 0.5) & bool(states_cfg.get("RHA", {}).get("exit_on_stall", False)))))
    exit_raw |= vt_low & (bars >= int(vcfg.get("low_move_confirm_bars", 2)))
    exit_raw |= st_issue & (bars >= int(scfg.get("exit_confirm_bars", 2)))
    out.loc[exit_raw, "price_expectation_exit_flag"] = 1

    # Quality penalty. This is conservative and does not overwrite stable_state.
    pe_score = pd.to_numeric(out["state_price_expectation_score"], errors="coerce").fillna(0.5).clip(0, 1)
    q = (base_q * (0.25 + 0.75 * pe_score)).clip(0, 1)
    # Directional no-trade cases get capped.
    q.loc[stable.eq("RHA") & (rha_stall > 0.5)] = np.minimum(q.loc[stable.eq("RHA") & (rha_stall > 0.5)], float(states_cfg.get("RHA", {}).get("stall_trading_quality_cap", 0.35)))
    q.loc[stable.eq("VT") & vt_mixed] = np.minimum(q.loc[stable.eq("VT") & vt_mixed], float(states_cfg.get("VT", {}).get("mixed_trading_quality_cap", 0.35)))
    out["price_expectation_trading_quality"] = q.clip(0, 1)

    if enabled:
        out["trading_trigger_quality"] = np.minimum(base_q, out["price_expectation_trading_quality"]).clip(0, 1)
        # Keep directional_state if present but append issue to no-trade reason.
        issue_text = out["state_price_expectation_issue"].astype(str).replace("", np.nan)
        old_reason = out.get("directional_no_trade_reason", pd.Series("", index=out.index)).astype(str)
        new_reason = old_reason.where(issue_text.isna(), (old_reason.where(old_reason.str.len() > 0, "") + np.where(old_reason.str.len() > 0, "; ", "") + issue_text.fillna("")))
        out["directional_no_trade_reason"] = new_reason.fillna("")
    return out
