from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def _safe_series(df: pd.DataFrame, col: str, default: Any = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_safe_series(df, col, default), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    return _num(df, col, 0.0) > 0.5


def _append_reason(base: pd.Series, addition: pd.Series) -> pd.Series:
    b = base.fillna("").astype(str)
    a = addition.fillna("").astype(str)
    out = b.copy()
    m = a.str.len() > 0
    out.loc[m & (b.str.len() == 0)] = a.loc[m]
    out.loc[m & (b.str.len() > 0)] = b.loc[m & (b.str.len() > 0)] + "; " + a.loc[m & (b.str.len() > 0)]
    return out


def _score_small(x: pd.Series, threshold: float) -> pd.Series:
    return (1.0 - (x / max(float(threshold), 1e-9)).clip(0, 1)).clip(0, 1)


def _score_large(x: pd.Series, threshold: float) -> pd.Series:
    return (x / max(float(threshold), 1e-9)).clip(0, 1)


def apply_price_expectation_compliance(df: pd.DataFrame, model_config: dict[str, Any]) -> pd.DataFrame:
    """V2.5 causal price-expectation compliance layer.

    It checks whether the current stable state's entry-to-current price behavior still matches
    the state's trading interpretation. It does not rewrite stable_state; it lowers trading quality
    and emits exit suggestions that are also mirrored inside the stability layer when the full
    pipeline is rerun.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    cfg = model_config.get("state_price_expectation", {}) if isinstance(model_config.get("state_price_expectation", {}), dict) else {}
    state_cfgs = cfg.get("states", {}) if isinstance(cfg.get("states", {}), dict) else cfg
    if not bool(cfg.get("enabled", True)):
        out["state_price_expectation_score"] = 1.0
        out["state_price_expectation_issue"] = "OK"
        out["price_expectation_exit_flag"] = 0
        out["price_expectation_exit_target"] = ""
        out["price_expectation_exit_reason"] = "disabled"
        out["price_expectation_trading_quality"] = _num(out, "trading_trigger_quality", 0.5).clip(0, 1)
        return out

    if "segment_id" not in out.columns:
        out["segment_id"] = out["stable_state"].astype(str).ne(out["stable_state"].astype(str).shift()).cumsum().astype(int)
    if "bars_since_state_entry" not in out.columns:
        out["bars_since_state_entry"] = out.groupby("segment_id", sort=False).cumcount() + 1
    if "state_entry_price" not in out.columns and "price" in out.columns:
        out["state_entry_price"] = out.groupby("segment_id", sort=False)["price"].transform("first")
    if "entry_to_current_return_bps" not in out.columns and {"price", "state_entry_price"}.issubset(out.columns):
        out["entry_to_current_return_bps"] = 10000.0 * np.log(pd.to_numeric(out["price"], errors="coerce") / pd.to_numeric(out["state_entry_price"], errors="coerce"))
    if "entry_to_current_abs_return_bps" not in out.columns:
        out["entry_to_current_abs_return_bps"] = _num(out, "entry_to_current_return_bps", 0.0).abs()
    if "entry_to_current_range_bps" not in out.columns and "price" in out.columns:
        p = pd.to_numeric(out["price"], errors="coerce")
        out["entry_to_current_range_bps"] = 10000.0 * np.log(p.groupby(out["segment_id"], sort=False).cummax() / p.groupby(out["segment_id"], sort=False).cummin())

    state = out["stable_state"].astype(str)
    bars = _num(out, "bars_since_state_entry", 1.0)
    abs_ret = _num(out, "entry_to_current_abs_return_bps", 0.0)
    rng = _num(out, "entry_to_current_range_bps", 0.0)
    ts6 = _num(out, "trend_strength_6h", 0.0).clip(0, 1)
    tc6 = _num(out, "trend_consistency_6h", 0.0).clip(0, 1)
    rv1 = _num(out, "realized_vol_1h_bps", 0.0)
    rv6 = _num(out, "realized_vol_6h_bps", 0.0)
    e_active = _num(out, "e_active", 0.0).clip(0, 1)
    e_base = _num(out, "e_baseline", 0.0).clip(0, 1)
    e_cas = _num(out, "e_cascade", 0.0).clip(0, 1)
    e_rej = _num(out, "e_rejection", 0.0).clip(0, 1)
    e_abs = _num(out, "e_absorption", 0.0).clip(0, 1)
    e_conf = _num(out, "e_conflict", 0.0).clip(0, 1)

    out["state_price_expectation_score"] = 1.0
    out["state_price_expectation_issue"] = "OK"
    out["price_expectation_exit_flag"] = 0
    out["price_expectation_exit_target"] = ""
    out["price_expectation_exit_reason"] = ""
    out["price_expectation_exit_pressure"] = 0.0
    out["price_expectation_raw_issue_flag"] = 0
    out["price_expectation_trading_quality"] = _num(out, "trading_trigger_quality", 0.5).clip(0, 1)
    for c in [
        "rc_price_drift_flag", "rc_range_break_flag", "rc_trend_exclusion_flag",
        "vt_low_move_flag", "vt_weak_trend_flag", "vt_mixed_direction_flag",
        "st_flat_flag", "st_weak_trend_flag", "st_too_extreme_flag",
        "hpem_price_expectation_fail_flag", "rha_price_expectation_fail_flag",
    ]:
        out[c] = 0

    # HPEM: price should continue in PLIE direction.
    c = state_cfgs.get("HPEM", {}) if isinstance(state_cfgs.get("HPEM", {}), dict) else {}
    m = state.eq("HPEM")
    if m.any():
        no_resp = _flag(out, "hpem_no_response_flag")
        decay = _flag(out, "hpem_decay_flag")
        adverse = _flag(out, "hpem_adverse_flag")
        comp = _num(out, "hpem_directional_compliance_score", 0.0).clip(0, 1)
        impact = _num(out, "hpem_price_impact_score", 0.0).clip(0, 1)
        give_penalty = 1 - _num(out, "hpem_giveback_ratio", 0.0).clip(0, 1)
        score = pd.concat([comp, impact, give_penalty], axis=1).mean(axis=1).clip(0, 1)
        score.loc[no_resp] = np.minimum(score.loc[no_resp], 0.25)
        score.loc[decay] = np.minimum(score.loc[decay], 0.30)
        score.loc[adverse] = np.minimum(score.loc[adverse], 0.20)
        fail = m & (no_resp | decay | adverse)
        out.loc[m, "state_price_expectation_score"] = score.loc[m]
        out.loc[fail, "hpem_price_expectation_fail_flag"] = 1
        out.loc[fail, "state_price_expectation_issue"] = "HPEM_PRICE_EXPECTATION_FAIL"
        out.loc[fail, "price_expectation_exit_flag"] = 1
        target = pd.Series("ST", index=out.index, dtype=object)
        target.loc[(e_rej > 0.35) | (e_abs > 0.55)] = "RHA"
        target.loc[(e_active > 0.55) & (e_cas < 0.45)] = "VT"
        target.loc[e_conf > 0.55] = "AMB"
        out.loc[fail, "price_expectation_exit_target"] = target.loc[fail]
        reason = pd.Series("HPEM no response / decay / adverse to PLIE", index=out.index, dtype=object)
        reason.loc[decay] = "HPEM PLIE-aligned move gave back"
        reason.loc[adverse] = "HPEM moved adversely against PLIE direction"
        out.loc[fail, "price_expectation_exit_reason"] = reason.loc[fail]
        out.loc[fail, "price_expectation_trading_quality"] = _num(out, "trading_trigger_quality", 0.5).loc[fail] * float(c.get("fail_quality_multiplier", 0.35))

    # RHA: reversal is tradable; stall is low quality; following PLIE invalidates RHA.
    c = state_cfgs.get("RHA", {}) if isinstance(state_cfgs.get("RHA", {}), dict) else {}
    m = state.eq("RHA")
    if m.any():
        invalid = _flag(out, "rha_invalid_follow_plie_flag")
        stall = _flag(out, "rha_stall_flag")
        rev = _flag(out, "rha_reversal_confirmed_flag")
        weak = m & (~invalid) & (~stall) & (~rev) & (bars >= float(c.get("weak_no_reversal_confirm_bars", 4)))
        comp = _num(out, "rha_directional_compliance_score", 0.0).clip(0, 1)
        score = comp.copy()
        score.loc[stall] = np.minimum(score.loc[stall], float(c.get("stall_score_cap", 0.48)))
        score.loc[weak] = np.minimum(score.loc[weak], float(c.get("weak_score_cap", 0.35)))
        score.loc[invalid] = np.minimum(score.loc[invalid], float(c.get("invalid_score_cap", 0.20)))
        out.loc[m, "state_price_expectation_score"] = score.loc[m]
        bad = m & (invalid | stall | weak)
        out.loc[bad, "rha_price_expectation_fail_flag"] = 1
        out.loc[stall, "state_price_expectation_issue"] = "RHA_STALL_NOT_DIRECTIONAL"
        out.loc[weak, "state_price_expectation_issue"] = "RHA_WEAK_NO_REVERSAL"
        out.loc[invalid, "state_price_expectation_issue"] = "RHA_INVALID_FOLLOWS_PLIE"
        hard = m & invalid
        out.loc[hard, "price_expectation_exit_flag"] = 1
        target = pd.Series("RC", index=out.index, dtype=object)
        target.loc[e_cas > 0.45] = "HPEM"
        target.loc[e_base > 0.45] = "ST"
        target.loc[e_conf > 0.55] = "AMB"
        out.loc[hard, "price_expectation_exit_target"] = target.loc[hard]
        out.loc[hard, "price_expectation_exit_reason"] = "RHA invalid: price follows PLIE pressure"
        mult = pd.Series(1.0, index=out.index)
        mult.loc[stall] = float(c.get("stall_quality_multiplier", 0.55))
        mult.loc[weak] = float(c.get("weak_quality_multiplier", 0.40))
        mult.loc[invalid] = float(c.get("invalid_quality_multiplier", 0.20))
        out.loc[bad, "price_expectation_trading_quality"] = _num(out, "trading_trigger_quality", 0.5).loc[bad] * mult.loc[bad]

    # RC: price change should be small.
    c = state_cfgs.get("RC", {}) if isinstance(state_cfgs.get("RC", {}), dict) else {}
    m = state.eq("RC")
    if m.any():
        drift = m & (abs_ret > float(c.get("max_entry_to_current_abs_return_bps", 80.0)))
        range_break = m & (rng > float(c.get("max_entry_to_current_range_bps", 140.0)))
        trend_ex = m & (ts6 > float(c.get("max_trend_strength_6h", 0.50))) & (tc6 > float(c.get("max_trend_consistency_6h", 0.58)))
        bad = drift | range_break | trend_ex
        out.loc[drift, "rc_price_drift_flag"] = 1
        out.loc[range_break, "rc_range_break_flag"] = 1
        out.loc[trend_ex, "rc_trend_exclusion_flag"] = 1
        max_ret = float(c.get("max_entry_to_current_abs_return_bps", 80.0)); max_rng = float(c.get("max_entry_to_current_range_bps", 140.0))
        score = pd.concat([_score_small(abs_ret, max_ret), _score_small(rng, max_rng), (1 - (ts6*tc6).clip(0,1))], axis=1).mean(axis=1).clip(0,1)
        out.loc[m, "state_price_expectation_score"] = score.loc[m]
        out.loc[bad, "state_price_expectation_issue"] = "RC_PRICE_TOO_LARGE_OR_TRENDING"
        out.loc[bad, "price_expectation_exit_flag"] = 1
        target = pd.Series("ST", index=out.index, dtype=object)
        target.loc[e_active > 0.45] = "VT"; target.loc[e_cas > 0.45] = "HPEM"; target.loc[e_conf > 0.55] = "AMB"
        out.loc[bad, "price_expectation_exit_target"] = target.loc[bad]
        out.loc[bad, "price_expectation_exit_reason"] = "RC price movement/range/trend exceeds consolidation expectation"
        out.loc[bad, "price_expectation_trading_quality"] = _num(out, "trading_trigger_quality", 0.5).loc[bad] * float(c.get("fail_quality_multiplier", 0.35))

    # VT: large move / range / volatility; direction can be mixed but then no directional trade.
    c = state_cfgs.get("VT", {}) if isinstance(state_cfgs.get("VT", {}), dict) else {}
    m = state.eq("VT")
    if m.any():
        low_move = m & (bars >= float(c.get("grace_bars", 2))) & (abs_ret < float(c.get("min_entry_to_current_abs_return_bps", 60.0))) & (rng < float(c.get("min_entry_to_current_range_bps", 120.0)))
        weak_vol = m & (rv1 < float(c.get("min_realized_vol_1h_bps", 45.0))) & (rv6 < float(c.get("min_realized_vol_6h_bps", 100.0)))
        mixed = m & _safe_series(out, "vt_direction_state", "").astype(str).eq("VT_MIXED")
        out.loc[low_move, "vt_low_move_flag"] = 1
        out.loc[weak_vol, "vt_weak_trend_flag"] = 1
        out.loc[mixed, "vt_mixed_direction_flag"] = 1
        min_ret=float(c.get("min_entry_to_current_abs_return_bps",60.0)); min_rng=float(c.get("min_entry_to_current_range_bps",120.0))
        score = pd.concat([_score_large(abs_ret,min_ret), _score_large(rng,min_rng), _num(out,"vt_direction_confidence",0.0).clip(0,1)], axis=1).mean(axis=1).clip(0,1)
        score.loc[mixed] = np.minimum(score.loc[mixed], float(c.get("mixed_score_cap",0.50)))
        score.loc[low_move | weak_vol] = np.minimum(score.loc[low_move | weak_vol], float(c.get("weak_score_cap",0.35)))
        out.loc[m, "state_price_expectation_score"] = score.loc[m]
        out.loc[mixed, "state_price_expectation_issue"] = "VT_MIXED_DIRECTION"
        out.loc[weak_vol, "state_price_expectation_issue"] = "VT_WEAK_VOLATILITY"
        out.loc[low_move, "state_price_expectation_issue"] = "VT_LOW_MOVE"
        hard = low_move | weak_vol
        out.loc[hard, "price_expectation_exit_flag"] = 1
        target = pd.Series("RC", index=out.index, dtype=object); target.loc[(e_base>0.35)&(ts6>0.35)]="ST"; target.loc[e_cas>0.55]="HPEM"; target.loc[e_conf>0.55]="AMB"
        out.loc[hard, "price_expectation_exit_target"] = target.loc[hard]
        out.loc[hard, "price_expectation_exit_reason"] = "VT has insufficient realized movement/volatility"
        out.loc[mixed | hard, "price_expectation_trading_quality"] = _num(out,"trading_trigger_quality",0.5).loc[mixed|hard] * np.where(mixed.loc[mixed|hard], float(c.get("mixed_quality_multiplier",0.45)), float(c.get("fail_quality_multiplier",0.35)))

    # ST: mild orderly trend; not flat, not extreme.
    c = state_cfgs.get("ST", {}) if isinstance(state_cfgs.get("ST", {}), dict) else {}
    m = state.eq("ST")
    if m.any():
        flat = m & (bars >= float(c.get("grace_bars",2))) & (abs_ret < float(c.get("min_entry_to_current_abs_return_bps",35.0))) & (rng < float(c.get("min_entry_to_current_range_bps",70.0)))
        weak = m & ((ts6 < float(c.get("min_trend_strength_6h",0.38))) | (tc6 < float(c.get("min_trend_consistency_6h",0.50))))
        extreme = m & ((abs_ret > float(c.get("max_entry_to_current_abs_return_bps",220.0))) | (rng > float(c.get("max_entry_to_current_range_bps",420.0))) | (rv1 > float(c.get("max_realized_vol_1h_bps",80.0))))
        out.loc[flat,"st_flat_flag"] = 1; out.loc[weak,"st_weak_trend_flag"] = 1; out.loc[extreme,"st_too_extreme_flag"] = 1
        score = pd.concat([_score_large(abs_ret,float(c.get("min_entry_to_current_abs_return_bps",35.0))), _score_large(rng,float(c.get("min_entry_to_current_range_bps",70.0))), ts6, tc6], axis=1).mean(axis=1).clip(0,1)
        score.loc[flat|weak] = np.minimum(score.loc[flat|weak],0.35); score.loc[extreme]=np.minimum(score.loc[extreme],0.40)
        out.loc[m,"state_price_expectation_score"] = score.loc[m]
        out.loc[flat,"state_price_expectation_issue"] = "ST_FLAT"; out.loc[weak,"state_price_expectation_issue"] = "ST_WEAK_TREND"; out.loc[extreme,"state_price_expectation_issue"] = "ST_TOO_EXTREME"
        bad = flat | weak | extreme
        out.loc[bad,"price_expectation_exit_flag"] = 1
        target = pd.Series("RC", index=out.index, dtype=object); target.loc[extreme & (e_active>0.45)]="VT"; target.loc[extreme & (e_cas>0.45)]="HPEM"; target.loc[e_conf>0.55]="AMB"
        out.loc[bad,"price_expectation_exit_target"] = target.loc[bad]
        out.loc[bad,"price_expectation_exit_reason"] = "ST price behavior is flat/weak trend/extreme"
        out.loc[bad,"price_expectation_trading_quality"] = _num(out,"trading_trigger_quality",0.5).loc[bad] * float(c.get("fail_quality_multiplier",0.40))

    pe_score_final = _num(out, "state_price_expectation_score", 1.0).clip(0,1)
    out["price_expectation_exit_pressure"] = (1.0 - pe_score_final).clip(0, 1)
    out.loc[_num(out, "price_expectation_exit_flag", 0.0) <= 0.5, "price_expectation_exit_pressure"] = 0.0
    out["price_expectation_raw_issue_flag"] = (_safe_series(out, "state_price_expectation_issue", "OK").astype(str) != "OK").astype(int)

    old_q = _num(out, "trading_trigger_quality", 0.5).clip(0,1)
    pe_q = _num(out, "price_expectation_trading_quality", old_q).clip(0,1)
    out["trading_trigger_quality"] = np.minimum(old_q, pe_q).clip(0,1)
    if "directional_trading_quality" in out.columns:
        out["directional_trading_quality"] = np.minimum(_num(out,"directional_trading_quality",0.5).clip(0,1), pe_q).clip(0,1)
    if "directional_no_trade_reason" in out.columns:
        out["directional_no_trade_reason"] = _append_reason(out["directional_no_trade_reason"], out["state_price_expectation_issue"].replace("OK", ""))
    return out

# Backward-compatible public alias.
apply_price_expectation = apply_price_expectation_compliance
