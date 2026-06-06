from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

UP_ACTIVE_LABELS = {
    "path_active_dominance_up",
    "path_normal_active_dominance_up",
    "path_mixed_active_breakout_up",
    "path_normal_mixed_active_breakout_up",
}
DOWN_ACTIVE_LABELS = {
    "path_active_dominance_down",
    "path_normal_active_dominance_down",
    "path_mixed_active_breakout_down",
    "path_normal_mixed_active_breakout_down",
}
REJECTION_LABELS = {"path_pressure_rejection", "path_reversal_takeover", "path_full_absorption_stall"}


def _safe_series(df: pd.DataFrame, col: str, default: Any = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_safe_series(df, col, default), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _sign_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.sign(x).astype(float)


def _path_active_dir_series(s: pd.Series) -> pd.Series:
    text = s.fillna("").astype(str)
    out = pd.Series(0.0, index=s.index)
    out.loc[text.isin(UP_ACTIVE_LABELS)] = 1.0
    out.loc[text.isin(DOWN_ACTIVE_LABELS)] = -1.0
    return out


def _direction_label_arr(prefix: str, direction: pd.Series) -> pd.Series:
    out = pd.Series(f"{prefix}_MIXED", index=direction.index, dtype=object)
    out.loc[direction > 0] = f"{prefix}_UP"
    out.loc[direction < 0] = f"{prefix}_DOWN"
    return out


def _segment_pressure_direction(df: pd.DataFrame) -> pd.Series:
    """First non-zero PLIE direction per stable-state segment, fallback to state pressure direction."""
    seg = df["segment_id"]
    plie = pd.to_numeric(_safe_series(df, "plie_direction", 0.0), errors="coerce").fillna(0.0)
    press = pd.to_numeric(_safe_series(df, "state_pressure_direction", 0.0), errors="coerce").fillna(0.0)
    tmp = pd.DataFrame({"segment_id": seg, "plie": plie, "press": press})

    def first_nonzero(g: pd.DataFrame) -> float:
        p = g["plie"]
        p = p[p.abs() > 0.5]
        if len(p):
            return float(np.sign(p.iloc[0]))
        q = g["press"]
        q = q[q.abs() > 0.5]
        if len(q):
            return float(np.sign(q.iloc[0]))
        return 0.0

    mapping = tmp.groupby("segment_id", sort=False).apply(first_nonzero)
    return seg.map(mapping).fillna(0.0).astype(float)


def _rolling_ratio_by_segment(flag: pd.Series, seg: pd.Series, window: int) -> pd.Series:
    return flag.astype(float).groupby(seg, sort=False).transform(lambda x: x.rolling(window, min_periods=1).mean())


def apply_directional_quality(df: pd.DataFrame, model_config: dict[str, Any]) -> pd.DataFrame:
    """Add causal directional lifecycle diagnostics after stable-state confirmation.

    This post-stability layer does not overwrite `stable_state`. It uses only information
    available from the current stable-state segment entry up to the current row: entry-to-current
    return/range, cumulative aligned MFE, and rolling recent direction ratios.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    cfg = model_config.get("directional_quality", {}) if isinstance(model_config.get("directional_quality", {}), dict) else {}
    if not bool(cfg.get("enabled", True)):
        out["directional_state"] = out.get("stable_state", "")
        out["directional_trading_quality"] = pd.to_numeric(_safe_series(out, "trading_trigger_quality", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
        return out

    hpem_cfg = cfg.get("HPEM", {}) if isinstance(cfg.get("HPEM", {}), dict) else {}
    rha_cfg = cfg.get("RHA", {}) if isinstance(cfg.get("RHA", {}), dict) else {}
    vt_cfg = cfg.get("VT", {}) if isinstance(cfg.get("VT", {}), dict) else {}
    st_cfg = cfg.get("ST", {}) if isinstance(cfg.get("ST", {}), dict) else {}

    if "stable_state" not in out.columns:
        raise ValueError("apply_directional_quality requires stable_state column")
    if "price" not in out.columns:
        raise ValueError("apply_directional_quality requires price column")

    out["price"] = pd.to_numeric(out["price"], errors="coerce")
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
    ret = pd.to_numeric(out["entry_to_current_return_bps"], errors="coerce").fillna(0.0)
    abs_ret = ret.abs()
    bars = pd.to_numeric(_safe_series(out, "bars_since_state_entry", 1), errors="coerce").fillna(1)
    base_quality = pd.to_numeric(_safe_series(out, "trading_trigger_quality", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
    pressure_ref = _segment_pressure_direction(out)
    out["pressure_direction_ref"] = pressure_ref

    # Defaults.
    out["directional_state"] = stable
    out["trade_direction"] = "NONE"
    out["direction_confidence"] = 0.0
    out["direction_source"] = "not_directional"
    out["directional_trading_quality"] = base_quality
    out["directional_no_trade_reason"] = ""
    out["directional_exit_trigger_flag"] = 0
    out["directional_exit_reason"] = ""
    out["directional_exit_target"] = ""

    numeric_defaults = [
        "hpem_expected_direction", "hpem_aligned_move_bps", "hpem_aligned_mfe_bps", "hpem_giveback_bps",
        "hpem_giveback_ratio", "hpem_directional_compliance_score", "rha_expected_direction",
        "rha_reversal_move_bps", "rha_reversal_mfe_bps", "rha_follow_plie_move_bps", "rha_reversal_score",
        "rha_directional_compliance_score", "vt_direction_score", "vt_direction_confidence",
        "vt_up_ratio_recent", "vt_down_ratio_recent", "st_direction_confidence",
    ]
    for c in numeric_defaults:
        out[c] = 0.0
    flag_defaults = [
        "hpem_no_response_flag", "hpem_decay_flag", "hpem_adverse_flag", "rha_stall_flag",
        "rha_reversal_confirmed_flag", "rha_invalid_follow_plie_flag", "vt_direction_flip_flag",
    ]
    for c in flag_defaults:
        out[c] = 0
    out["rha_subtype"] = ""
    out["vt_direction_state"] = ""
    out["st_direction_state"] = ""

    # ---------------- HPEM directional compliance ----------------
    m = stable.eq("HPEM")
    if m.any():
        expected = pressure_ref.copy()
        # If no PLIE pressure is available, fallback to sign of causal entry-to-current return.
        expected.loc[m & expected.eq(0)] = np.sign(ret.loc[m & expected.eq(0)]).replace(0, 0.0)
        aligned = expected * ret
        aligned_m = aligned.where(m, 0.0)
        mfe = aligned_m.groupby(seg, sort=False).cummax().clip(lower=0.0)
        giveback = (mfe - aligned_m).clip(lower=0.0)
        ratio = (giveback / (mfe.abs() + 1e-9)).clip(0, 9)
        impact = pd.to_numeric(_safe_series(out, "hpem_price_impact_score", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        shock_aligned = pd.to_numeric(_safe_series(out, "e_shock_aligned_plie", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        min_mfe = float(hpem_cfg.get("min_aligned_mfe_bps", 80.0))
        min_impact = float(hpem_cfg.get("min_price_impact_score", 0.45))
        grace = int(hpem_cfg.get("grace_bars", 2))
        give_abs = float(hpem_cfg.get("giveback_abs_bps", 120.0))
        give_ratio = float(hpem_cfg.get("giveback_ratio", 0.55))
        adverse = float(hpem_cfg.get("adverse_move_bps", 60.0))
        path6 = _safe_series(out, "path_label_6h", "").fillna("").astype(str)
        rejection_like = path6.isin(REJECTION_LABELS)
        no_resp = m & (bars >= grace) & (mfe < min_mfe) & (impact < min_impact)
        decay = m & (mfe >= min_mfe) & (giveback >= give_abs) & (ratio >= give_ratio)
        adverse_flag = m & (aligned <= -adverse)
        compliance = pd.concat([(mfe / max(min_mfe, 1e-9)).clip(0, 1), impact, shock_aligned], axis=1).max(axis=1)
        compliance = (compliance * (1 - 0.65 * ratio.clip(0, 1))).clip(0, 1)
        q = (base_quality * (0.35 + 0.65 * compliance)).clip(0, 1)
        weak = m & (no_resp | decay | adverse_flag | (rejection_like & (impact < min_impact)))
        direction = pd.Series("MIXED", index=out.index, dtype=object)
        direction.loc[expected > 0] = "UP"
        direction.loc[expected < 0] = "DOWN"
        out.loc[m, "hpem_expected_direction"] = expected.loc[m]
        out.loc[m, "hpem_aligned_move_bps"] = aligned.loc[m]
        out.loc[m, "hpem_aligned_mfe_bps"] = mfe.loc[m]
        out.loc[m, "hpem_giveback_bps"] = giveback.loc[m]
        out.loc[m, "hpem_giveback_ratio"] = ratio.loc[m]
        out.loc[m, "hpem_directional_compliance_score"] = compliance.loc[m]
        out.loc[no_resp, "hpem_no_response_flag"] = 1
        out.loc[decay, "hpem_decay_flag"] = 1
        out.loc[adverse_flag, "hpem_adverse_flag"] = 1
        out.loc[m, "directional_state"] = "HPEM_" + direction.loc[m]
        out.loc[m, "trade_direction"] = direction.loc[m]
        out.loc[m, "direction_confidence"] = compliance.loc[m]
        out.loc[m, "direction_source"] = "HPEM direction = entry PLIE pressure; quality = aligned MFE / price impact / giveback"
        out.loc[m, "directional_trading_quality"] = q.loc[m]
        out.loc[weak, "directional_state"] = "WEAK_HPEM"
        out.loc[weak, "directional_trading_quality"] = np.minimum(q.loc[weak], 0.32)
        out.loc[no_resp, "directional_no_trade_reason"] = "HPEM no price response after grace bars"
        out.loc[decay, "directional_no_trade_reason"] = "HPEM giveback/decay exceeded threshold"
        out.loc[adverse_flag, "directional_no_trade_reason"] = "HPEM adverse move against expected PLIE direction"
        exit_cond = m & (decay | adverse_flag | (no_resp & rejection_like))
        out.loc[exit_cond, "directional_exit_trigger_flag"] = 1
        out.loc[decay, "directional_exit_reason"] = "HPEM aligned MFE gave back beyond threshold"
        out.loc[adverse_flag, "directional_exit_reason"] = "HPEM adverse move against PLIE direction"
        out.loc[no_resp & rejection_like, "directional_exit_reason"] = "HPEM no-response with short-window rejection label"
        out.loc[exit_cond & rejection_like, "directional_exit_target"] = "RHA"
        out.loc[exit_cond & ~rejection_like, "directional_exit_target"] = "VT_OR_AMB"

    # ---------------- RHA subtype and invalidity ----------------
    m = stable.eq("RHA")
    if m.any():
        expected = -pressure_ref
        reversal = expected * ret
        reversal_m = reversal.where(m, 0.0)
        mfe = reversal_m.groupby(seg, sort=False).cummax().clip(lower=0.0)
        follows = pressure_ref * ret
        e_abs = pd.to_numeric(_safe_series(out, "e_absorption", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        e_rej = pd.to_numeric(_safe_series(out, "e_rejection", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        e_rha = pd.to_numeric(_safe_series(out, "e_rha_strict", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        e_cas = pd.to_numeric(_safe_series(out, "e_cascade", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        e_base = pd.to_numeric(_safe_series(out, "e_baseline", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        min_rev = float(rha_cfg.get("reversal_min_mfe_bps", 80.0))
        min_score = float(rha_cfg.get("reversal_min_score", 0.60))
        stall_max = float(rha_cfg.get("stall_max_abs_move_bps", 60.0))
        stall_abs = float(rha_cfg.get("stall_requires_absorption", 0.55))
        invalid_bps = float(rha_cfg.get("invalid_if_price_follows_plie_bps", 80.0))
        rev_score = pd.concat([(mfe / max(min_rev, 1e-9)).clip(0, 1), e_rej, e_rha], axis=1).mean(axis=1).clip(0, 1)
        stall = m & (abs_ret <= stall_max) & (e_abs >= stall_abs) & (e_cas < 0.55)
        invalid = m & (follows >= invalid_bps) & ((e_cas > 0.45) | (e_base > 0.45))
        reversal_confirmed = m & (mfe >= min_rev) & (rev_score >= min_score)
        subtype = pd.Series("RHA_WEAK", index=out.index, dtype=object)
        subtype.loc[stall] = "RHA_STALL"
        subtype.loc[reversal_confirmed & (expected > 0)] = "RHA_REVERSAL_UP"
        subtype.loc[reversal_confirmed & (expected < 0)] = "RHA_REVERSAL_DOWN"
        subtype.loc[invalid] = "INVALID_RHA_FOLLOWS_PLIE"
        q = base_quality.copy()
        q.loc[stall] *= float(rha_cfg.get("stall_quality_multiplier", 0.65))
        q.loc[m & subtype.eq("RHA_WEAK")] *= float(rha_cfg.get("weak_quality_multiplier", 0.45))
        q.loc[invalid] *= float(rha_cfg.get("invalid_quality_multiplier", 0.20))
        out.loc[m, "rha_expected_direction"] = expected.loc[m]
        out.loc[m, "rha_reversal_move_bps"] = reversal.loc[m]
        out.loc[m, "rha_reversal_mfe_bps"] = mfe.loc[m]
        out.loc[m, "rha_follow_plie_move_bps"] = follows.loc[m]
        out.loc[m, "rha_reversal_score"] = rev_score.loc[m]
        out.loc[reversal_confirmed, "rha_reversal_confirmed_flag"] = 1
        out.loc[stall, "rha_stall_flag"] = 1
        out.loc[invalid, "rha_invalid_follow_plie_flag"] = 1
        comp = pd.concat([rev_score, e_abs, e_rej], axis=1).mean(axis=1).clip(0, 1)
        out.loc[m, "rha_directional_compliance_score"] = comp.loc[m]
        out.loc[m, "rha_subtype"] = subtype.loc[m]
        out.loc[m, "directional_state"] = subtype.loc[m]
        out.loc[m & (expected > 0), "trade_direction"] = "UP"
        out.loc[m & (expected < 0), "trade_direction"] = "DOWN"
        out.loc[m & (expected == 0), "trade_direction"] = "MIXED"
        out.loc[m, "direction_confidence"] = comp.loc[m]
        out.loc[m, "direction_source"] = "RHA direction = -entry PLIE; subtype separates reversal/stall/invalid"
        out.loc[m, "directional_trading_quality"] = q.loc[m].clip(0, 1)
        out.loc[invalid, "directional_no_trade_reason"] = "RHA invalid: price follows PLIE with cascade/baseline"
        out.loc[m & subtype.eq("RHA_WEAK"), "directional_no_trade_reason"] = "RHA weak: no reversal/stall confirmation"
        out.loc[stall, "directional_no_trade_reason"] = "RHA stall/high absorption without directional reversal"
        out.loc[invalid, "directional_exit_trigger_flag"] = 1
        out.loc[invalid, "directional_exit_reason"] = "RHA invalidated by price following PLIE pressure"
        out.loc[invalid & (e_cas > 0.45), "directional_exit_target"] = "HPEM"
        out.loc[invalid & ~(e_cas > 0.45), "directional_exit_target"] = "ST_OR_HPEM"

    # ---------------- VT direction layer ----------------
    m = stable.eq("VT")
    if m.any():
        td6 = _sign_series(_safe_series(out, "trend_direction_6h", 0.0))
        td6 = td6.where(td6.abs() > 0, _sign_series(_safe_series(out, "past_return_6h_bps", 0.0)))
        td1 = _sign_series(_safe_series(out, "trend_direction_1h", 0.0))
        td1 = td1.where(td1.abs() > 0, _sign_series(_safe_series(out, "past_return_1h_bps", 0.0)))
        ret3 = _sign_series(_safe_series(out, "past_return_3h_bps", 0.0))
        p6 = _path_active_dir_series(_safe_series(out, "path_label_6h", ""))
        p12 = _path_active_dir_series(_safe_series(out, "path_label_12h", ""))
        p24 = _path_active_dir_series(_safe_series(out, "path_label_24h", ""))
        score = (0.25 * td6 + 0.20 * ret3 + 0.15 * td1 + 0.20 * p6 + 0.10 * p12 + 0.10 * p24).clip(-1, 1)
        tr6 = pd.to_numeric(_safe_series(out, "trend_strength_6h", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        min_tr = float(vt_cfg.get("min_trend_strength_6h", 0.42))
        up_thr = float(vt_cfg.get("up_threshold", 0.30))
        dn_thr = float(vt_cfg.get("down_threshold", -0.30))
        conf = (score.abs() * (0.55 + 0.45 * (tr6 / max(min_tr, 1e-9)).clip(0, 1))).clip(0, 1)
        vt_state = pd.Series("VT_MIXED", index=out.index, dtype=object)
        vt_state.loc[(score >= up_thr) & (tr6 >= min_tr)] = "VT_UP"
        vt_state.loc[(score <= dn_thr) & (tr6 >= min_tr)] = "VT_DOWN"
        window = int(vt_cfg.get("mixed_window_bars", 6))
        ratio_thr = float(vt_cfg.get("mixed_if_up_and_down_ratio_gt", 0.25))
        up_ratio = _rolling_ratio_by_segment(vt_state.eq("VT_UP"), seg, window)
        dn_ratio = _rolling_ratio_by_segment(vt_state.eq("VT_DOWN"), seg, window)
        mixed = m & (up_ratio > ratio_thr) & (dn_ratio > ratio_thr)
        vt_state.loc[mixed] = "VT_MIXED"
        q = (base_quality * np.where(vt_state.eq("VT_MIXED"), float(vt_cfg.get("mixed_quality_multiplier", 0.35)), (0.35 + 0.65 * conf))).clip(0, 1)
        out.loc[m, "vt_direction_score"] = score.loc[m]
        out.loc[m, "vt_direction_state"] = vt_state.loc[m]
        out.loc[m, "vt_direction_confidence"] = conf.loc[m]
        out.loc[m, "vt_up_ratio_recent"] = up_ratio.loc[m]
        out.loc[m, "vt_down_ratio_recent"] = dn_ratio.loc[m]
        out.loc[mixed, "vt_direction_flip_flag"] = 1
        out.loc[m, "directional_state"] = vt_state.loc[m]
        out.loc[m & vt_state.eq("VT_UP"), "trade_direction"] = "UP"
        out.loc[m & vt_state.eq("VT_DOWN"), "trade_direction"] = "DOWN"
        out.loc[m & vt_state.eq("VT_MIXED"), "trade_direction"] = "MIXED"
        out.loc[m, "direction_confidence"] = conf.loc[m]
        out.loc[m, "direction_source"] = "VT direction = trend_direction/past_return/path_active_dominance composite"
        out.loc[m, "directional_trading_quality"] = q.loc[m]
        out.loc[m & vt_state.eq("VT_MIXED"), "directional_no_trade_reason"] = "VT mixed direction; no single directional trade"

    # ---------------- ST trend direction ----------------
    m = stable.eq("ST")
    if m.any():
        td6 = _sign_series(_safe_series(out, "trend_direction_6h", 0.0))
        td6 = td6.where(td6.abs() > 0, _sign_series(_safe_series(out, "past_return_6h_bps", 0.0)))
        tr6 = pd.to_numeric(_safe_series(out, "trend_strength_6h", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        tc6 = pd.to_numeric(_safe_series(out, "trend_consistency_6h", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        conf = (0.55 * tr6 + 0.45 * tc6).clip(0, 1)
        st_state = pd.Series("ST_MIXED", index=out.index, dtype=object)
        st_state.loc[td6 > 0] = "ST_UP"
        st_state.loc[td6 < 0] = "ST_DOWN"
        weak = m & ((tr6 < float(st_cfg.get("min_trend_strength_6h", 0.38))) | (tc6 < float(st_cfg.get("min_trend_consistency_6h", 0.50))))
        st_state.loc[weak] = "WEAK_ST"
        q = (base_quality * (0.40 + 0.60 * conf)).clip(0, 1)
        q.loc[weak] *= 0.45
        out.loc[m, "st_direction_state"] = st_state.loc[m]
        out.loc[m, "st_direction_confidence"] = conf.loc[m]
        out.loc[m, "directional_state"] = st_state.loc[m]
        out.loc[m & st_state.eq("ST_UP"), "trade_direction"] = "UP"
        out.loc[m & st_state.eq("ST_DOWN"), "trade_direction"] = "DOWN"
        out.loc[m & st_state.eq("ST_MIXED"), "trade_direction"] = "MIXED"
        out.loc[m, "direction_confidence"] = conf.loc[m]
        out.loc[m, "direction_source"] = "ST direction = 6h trend direction with strength/consistency confirmation"
        out.loc[m, "directional_trading_quality"] = q.loc[m]
        out.loc[weak, "directional_no_trade_reason"] = "ST weak trend direction/consistency"

    # ---------------- Non-directional states ----------------
    m = stable.isin(["RC", "AMB"])
    if m.any():
        out.loc[m, "directional_state"] = stable.loc[m] + "_NO_TRADE"
        out.loc[m, "trade_direction"] = "NO_TRADE"
        out.loc[m, "directional_trading_quality"] = np.minimum(base_quality.loc[m], 0.35)
        out.loc[m, "directional_no_trade_reason"] = stable.loc[m] + " is non-directional by definition"

    out["base_trading_trigger_quality"] = base_quality
    out["base_trading_trigger_state"] = _safe_series(out, "trading_trigger_state", stable)
    out["directional_trading_quality"] = pd.to_numeric(out["directional_trading_quality"], errors="coerce").fillna(base_quality).clip(0, 1)
    out["trading_trigger_quality"] = np.minimum(base_quality, out["directional_trading_quality"]).clip(0, 1)
    out["trading_trigger_state"] = out["directional_state"].astype(str)
    return out
