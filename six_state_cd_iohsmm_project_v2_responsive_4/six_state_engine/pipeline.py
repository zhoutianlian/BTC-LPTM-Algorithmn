from __future__ import annotations
from pathlib import Path
from typing import Any
import shutil
import pandas as pd

from .config import load_project_config
from .data_loader import InputZipLoader
from .canonical import CanonicalFrameBuilder
from .validation import write_no_leakage_report
from .evidence import EvidenceBuilder
from .filter import CDIOHSMMFilter
from .stability import ProductionStabilityLayer
from .segment_quality import apply_state_confirmation, save_segment_quality_outputs
from .directional_quality import apply_directional_quality
from .price_expectation import apply_price_expectation
from .evaluation import save_all_reports
from .visualization import build_dashboard
from .utils import setup_logger


def select_state_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "time", "price", "split",
        "path_context_6h", "path_label_6h", "path_context_12h", "path_label_12h",
        "path_context_24h", "path_label_24h", "path_context_48h", "path_label_48h",
        "plie_direction", "plie_main_bps", "plie_reliability", "plie_intensity", "plie_phase",
        "plie_strong_entry", "plie_transition_type", "plie_transition_severity",
        "hmm_state", "hmm_conf", "liq_entropy",
        "realized_vol_1h_bps", "realized_vol_6h_bps", "range_compression_1h", "range_compression_6h",
        "trend_strength_1h", "trend_direction_1h", "trend_strength_6h", "trend_consistency_6h",
        "max_jump_z_1h", "jump_proxy_1h", "jump_proxy_6h", "price_missing_ratio_1h",
        "price_gap_flag_1h", "price_outlier_flag_1h",
        "candidate_state", "candidate_confidence", "candidate_transition_probability",
        "belief_gap", "belief_entropy", "expected_state_age",
        "stable_state", "stable_state_id", "stable_state_duration", "stable_switch_flag",
        "from_state", "transition_reason", "top_evidence_features",
        "state_invalidity_score", "state_invalidity_reason", "exit_pressure_current_state",
        "fast_transition_flag", "fast_transition_reason", "exit_pressure_trigger_flag",
        "delayed_hold_flag", "effective_tau_in", "effective_belief_gap_min", "effective_ewma_lambda",
        "segment_id", "bars_since_state_entry", "state_entry_time", "state_entry_price",
        "entry_to_current_return_bps", "entry_to_current_abs_return_bps", "entry_to_current_range_bps",
        "entry_to_current_mfe_bps", "entry_to_current_mae_bps", "state_confirmation_score",
        "state_confirmation_status", "state_confirmed_flag", "state_pending_confirmation_flag",
        "weak_state_flag", "trading_trigger_quality", "trading_trigger_state", "state_quality_issue",
        "hpem_price_impact_score", "vt_movement_confirmation_score", "st_movement_confirmation_score",
        "base_trading_trigger_state", "base_trading_trigger_quality",
        "pressure_direction_ref", "directional_state", "trade_direction", "direction_confidence",
        "direction_source", "directional_trading_quality", "directional_no_trade_reason",
        "directional_exit_trigger_flag", "directional_exit_reason", "directional_exit_target",
        "hpem_expected_direction", "hpem_aligned_move_bps", "hpem_aligned_mfe_bps",
        "hpem_giveback_bps", "hpem_giveback_ratio", "hpem_directional_compliance_score",
        "hpem_no_response_flag", "hpem_decay_flag", "hpem_adverse_flag",
        "rha_expected_direction", "rha_reversal_move_bps", "rha_reversal_mfe_bps",
        "rha_follow_plie_move_bps", "rha_reversal_score", "rha_directional_compliance_score",
        "rha_stall_flag", "rha_reversal_confirmed_flag", "rha_invalid_follow_plie_flag",
        "rha_subtype", "vt_direction_score", "vt_direction_state", "vt_direction_confidence",
        "vt_up_ratio_recent", "vt_down_ratio_recent", "vt_direction_flip_flag",
        "st_direction_state", "st_direction_confidence",
        "price_expectation_live_score", "price_expectation_live_issue",
        "price_expectation_live_exit_flag", "price_expectation_live_exit_target",
        "price_expectation_live_reason", "price_expectation_invalidity_score",
        "price_expectation_override_flag",
        "state_price_expectation_score", "state_price_expectation_issue",
        "price_expectation_raw_issue_flag", "price_expectation_exit_flag",
        "price_expectation_exit_target", "price_expectation_exit_reason",
        "price_expectation_exit_pressure", "price_expectation_trading_quality",
        "price_expectation_issue_consecutive_count",
        "rc_price_drift_flag", "rc_range_break_flag", "rc_trend_exclusion_flag",
        "vt_low_move_flag", "vt_mixed_direction_flag", "vt_weak_trend_flag",
        "st_flat_flag", "st_weak_trend_flag", "st_too_extreme_flag",
        "hpem_price_expectation_fail_flag", "rha_price_expectation_fail_flag",
        "execution_state", "execution_trade_direction", "execution_quality", "execution_no_trade_reason",
        "pre_price_expectation_trading_quality",
    ]
    belief_cols = [c for c in df.columns if c.startswith("b_") or c.startswith("stable_b_")]
    evidence_cols = [c for c in df.columns if c.startswith("e_")]
    score_cols = [c for c in df.columns if c.startswith("score_") or c.startswith("prior_")]
    quality_cols = [
        "data_gap_flag", "source_stale_flag", "market_response_stale_flag",
        "path_stale_flag", "price_context_stale_flag", "missing_rate_row", "critical_missing_flag"
    ]
    cols = [c for c in base_cols + belief_cols + evidence_cols + score_cols + quality_cols if c in df.columns]
    # preserve order and uniqueness
    seen = set()
    final_cols = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            final_cols.append(c)
    return df[final_cols].copy()


def run_pipeline(input_zip: str | Path, output_dir: str | Path, config_dir: str | Path | None = None) -> dict[str, Any]:
    logger = setup_logger()
    project_root = Path(__file__).resolve().parents[1]
    config_dir = Path(config_dir) if config_dir else project_root / "configs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_project_config(config_dir)
    feature_contract = cfg["feature_contract"]
    model_config = cfg["model"]

    logger.info("Starting Six-State CD-IOHSMM pipeline")
    logger.info("Input zip: %s", input_zip)
    logger.info("Output dir: %s", output_dir)

    loader = InputZipLoader(input_zip, output_dir, feature_contract)
    roles = loader.match_roles()

    canonical_builder = CanonicalFrameBuilder(feature_contract)
    canonical, leakage_report = canonical_builder.build(roles)
    write_no_leakage_report(output_dir / "no_leakage_report.json", leakage_report)
    canonical.to_csv(output_dir / "canonical_features.csv", index=False)
    logger.info("Canonical frame rows=%d cols=%d", len(canonical), len(canonical.columns))

    evidence_builder = EvidenceBuilder(model_config).fit(canonical)
    evidence_df = evidence_builder.transform(canonical)
    evidence_cols = [c for c in evidence_df.columns if c.startswith("e_") or c.startswith("score_") or c.startswith("prior_")]
    evidence_df[["time", "price"] + evidence_cols].to_csv(output_dir / "semantic_evidence.csv", index=False)
    logger.info("Semantic evidence generated")

    filt = CDIOHSMMFilter(model_config)
    filtered = filt.filter(evidence_df)
    logger.info("Filtered CD-IOHSMM belief generated")

    stability = ProductionStabilityLayer(model_config)
    states = stability.apply(filtered)
    logger.info("Production stability layer applied")

    states = apply_state_confirmation(states, model_config)
    logger.info("Causal state confirmation / trading-trigger quality layer applied")

    states = apply_directional_quality(states, model_config)
    logger.info("Directional compliance / VT direction / directional trading quality layer applied")

    states = apply_price_expectation(states, model_config)
    logger.info("Price-expectation compliance / state-behavior quality layer applied")

    state_out = select_state_output_columns(states)
    state_out.to_csv(output_dir / "state_timeseries.csv", index=False)
    # Full debug output is useful but can be very large. It is configurable to keep
    # production runs and dashboard regeneration responsive.
    output_cfg = model_config.get("output", {}) if isinstance(model_config.get("output", {}), dict) else {}
    if bool(output_cfg.get("write_full_debug", False)):
        states.to_csv(output_dir / "state_timeseries_full_debug.csv", index=False)

    report = save_all_reports(output_dir, states, leakage_report.to_dict(), model_config)
    logger.info("Reports generated")

    segment_quality_summary = save_segment_quality_outputs(output_dir, states, model_config)
    logger.info("Segment quality audit generated: %d segments", segment_quality_summary.get("segments", 0) if isinstance(segment_quality_summary, dict) else 0)

    build_dashboard(states, output_dir / "six_state_dashboard.html")
    logger.info("Dashboard generated: %s", output_dir / "six_state_dashboard.html")

    # Copy configs into output for reproducibility
    cfg_out = output_dir / "used_configs"
    cfg_out.mkdir(exist_ok=True)
    for p in Path(config_dir).glob("*.yaml"):
        shutil.copy2(p, cfg_out / p.name)

    return {
        "output_dir": str(output_dir),
        "rows": int(len(states)),
        "no_leakage_passed": bool(leakage_report.passed),
        "state_distribution": report.get("state_distribution", {}),
        "segment_quality_summary": segment_quality_summary,
        "dashboard": str(output_dir / "six_state_dashboard.html"),
    }
