from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .constants import STATES, STATE_DEFINITIONS
from .utils import write_json
from .segment_quality import save_segment_quality_outputs


CORE_STAT_COLUMNS = [
    "e_directional_pressure", "e_pressure_strength", "e_baseline", "e_cascade",
    "e_absorption", "e_rejection", "e_rha_strict", "e_active", "e_quiet",
    "e_realized_vol", "e_volatility", "e_jump", "e_trend_strength",
    "e_range_compression", "e_low_vol", "e_low_trend",
    "e_conflict", "e_quality_bad", "e_cross_window_conflict",
    "plie_reliability", "plie_intensity", "hmm_conf", "liq_entropy",
    "realized_vol_6h_bps", "range_compression_6h", "trend_strength_6h",
    "trend_consistency_6h", "jump_proxy_6h",
    "path_absorption_score_6h", "path_absorption_score_12h", "path_absorption_score_24h", "path_absorption_score_48h",
    "path_pressure_rejection_score_6h", "path_pressure_rejection_score_12h", "path_pressure_rejection_score_24h", "path_pressure_rejection_score_48h",
    "path_active_dominance_score_6h", "path_active_dominance_score_12h", "path_active_dominance_score_24h", "path_active_dominance_score_48h",
    "path_cascade_score_6h", "path_cascade_score_12h", "path_cascade_score_24h", "path_cascade_score_48h",
    "path_data_quality_6h", "path_data_quality_12h", "path_data_quality_24h", "path_data_quality_48h",
    "path_signal_clarity_6h", "path_signal_clarity_12h", "path_signal_clarity_24h", "path_signal_clarity_48h",
]


def state_definitions_frame() -> pd.DataFrame:
    rows = []
    for s in STATES:
        d = STATE_DEFINITIONS[s]
        rows.append({"state_id": d["id"], "state": s, "name": d["name"], "cn": d["cn"], "meaning": d["meaning"]})
    return pd.DataFrame(rows)


def compute_state_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CORE_STAT_COLUMNS if c in df.columns]
    rows = []
    total = len(df)
    for state, g in df.groupby("stable_state"):
        row = {"state": state, "count": int(len(g)), "proportion": float(len(g) / max(total, 1))}
        for c in cols:
            x = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(x.mean()) if x.notna().any() else np.nan
            row[f"{c}_median"] = float(x.median()) if x.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("state")


def compute_duration_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    change = df["stable_state"].ne(df["stable_state"].shift(1)).cumsum()
    rows = []
    for _, g in df.groupby(change):
        state = g["stable_state"].iloc[0]
        rows.append({
            "state": state,
            "start_time": g["time"].iloc[0],
            "end_time": g["time"].iloc[-1],
            "n_steps": int(len(g)),
            "start_price": float(g["price"].iloc[0]) if "price" in g else np.nan,
            "end_price": float(g["price"].iloc[-1]) if "price" in g else np.nan,
            "mean_confidence": float(g["candidate_confidence"].mean()) if "candidate_confidence" in g else np.nan,
            "mean_belief_gap": float(g["belief_gap"].mean()) if "belief_gap" in g else np.nan,
        })
    return pd.DataFrame(rows)


def typical_scenarios(df: pd.DataFrame, per_state: int = 8) -> pd.DataFrame:
    rows = []
    for state in STATES:
        b_col = f"stable_b_{state}"
        if b_col not in df.columns:
            continue
        g = df[df["stable_state"] == state].copy()
        if g.empty:
            continue
        g["_rank"] = pd.to_numeric(g[b_col], errors="coerce").fillna(0) + 0.1 * pd.to_numeric(g["belief_gap"], errors="coerce").fillna(0)
        pick = g.sort_values("_rank", ascending=False).head(per_state)
        keep = [
            "time", "price", "split", "stable_state", b_col, "candidate_state",
            "belief_gap", "path_context_6h", "path_label_6h", "path_context_12h", "path_label_12h",
            "path_context_6h", "path_label_6h", "path_context_12h", "path_label_12h",
        "path_context_24h", "path_label_24h", "path_context_48h", "path_label_48h", "path_context_48h", "path_label_48h",
            "plie_direction", "plie_reliability", "plie_intensity",
            "e_cascade", "e_rejection", "e_rha_strict", "e_absorption", "e_active",
            "e_quiet", "e_low_vol", "e_range_compression", "e_trend_strength",
            "e_conflict", "e_quality_bad", "top_evidence_features",
        ]
        for _, r in pick.iterrows():
            rows.append({c: r.get(c, np.nan) for c in keep if c in pick.columns or c in df.columns})
    return pd.DataFrame(rows)


def transition_events(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.get("stable_switch_flag", 0) == 1].copy()
    keep = [
        "time", "price", "from_state", "stable_state", "candidate_state",
        "candidate_transition_probability", "stable_state_duration",
        "belief_gap", "transition_reason", "top_evidence_features",
        "path_context_6h", "path_label_6h", "path_context_12h", "path_label_12h",
        "path_context_24h", "path_label_24h", "path_context_48h", "path_label_48h",
    ]
    return g[[c for c in keep if c in g.columns]]


def scenario_review(df: pd.DataFrame, model_config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    windows = model_config.get("validation", {}).get("scenario_windows", {})
    for name, spec in windows.items():
        start = pd.to_datetime(spec["start"])
        end = pd.to_datetime(spec["end"])
        sub = df[(df["time"] >= start) & (df["time"] <= end)].copy()
        if sub.empty:
            out[name] = {"available": False, "note": spec.get("note", ""), "message": "No rows in this window."}
            continue
        state_counts = sub["stable_state"].value_counts(normalize=True).to_dict()
        primary = max(state_counts, key=state_counts.get)
        out[name] = {
            "available": True,
            "note": spec.get("note", ""),
            "start": str(start),
            "end": str(end),
            "rows": int(len(sub)),
            "primary_state": primary,
            "state_distribution": {k: float(v) for k, v in state_counts.items()},
            "expected_primary": spec.get("expected_primary", []),
            "pass_expectation": primary in spec.get("expected_primary", []),
            "mean_beliefs": {s: float(sub[f"b_{s}"].mean()) for s in STATES if f"b_{s}" in sub},
            "path_context_top_6h": sub["path_context_6h"].value_counts().head(5).to_dict() if "path_context_6h" in sub else {},
            "path_label_top_6h": sub["path_label_6h"].value_counts().head(5).to_dict() if "path_label_6h" in sub else {},
            "path_context_top_24h": sub["path_context_24h"].value_counts().head(5).to_dict() if "path_context_24h" in sub else {},
            "path_label_top_24h": sub["path_label_24h"].value_counts().head(5).to_dict() if "path_label_24h" in sub else {},
        }
    return out


def responsiveness_diagnostics(df: pd.DataFrame, model_config: dict[str, Any]) -> dict[str, Any]:
    """Diagnostics for v2.2 responsiveness mechanisms.

    These are audit metrics, not training objectives. They help verify whether price shocks
    and exit pressure reduce stale-state holding without creating uncontrolled churn.
    """
    if df.empty:
        return {}
    resp = model_config.get("responsiveness", {}) if isinstance(model_config.get("responsiveness", {}), dict) else {}
    shock_cfg = resp.get("price_shock", {}) if isinstance(resp.get("price_shock", {}), dict) else {}
    shock_thr = float(shock_cfg.get("min_short_impulse_1h", 0.45))
    valid = pd.to_numeric(df.get("e_shock_data_valid", 0), errors="coerce").fillna(0) >= 0.5
    impulse = pd.to_numeric(df.get("e_short_impulse_1h", 0), errors="coerce").fillna(0)
    shock = (impulse >= shock_thr) & valid
    switch = pd.to_numeric(df.get("stable_switch_flag", 0), errors="coerce").fillna(0).astype(bool)

    def within_k(mask: pd.Series, k: int) -> float:
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            return 0.0
        sw = switch.to_numpy()
        ok = 0
        for i in idx:
            j1 = min(len(sw), i + k + 1)
            if sw[i:j1].any():
                ok += 1
        return float(ok / len(idx))

    mismatch = df.get("candidate_state", pd.Series(index=df.index, dtype=object)).astype(str).ne(df.get("stable_state", pd.Series(index=df.index, dtype=object)).astype(str))
    return {
        "shock_event_count": int(shock.sum()),
        "shock_event_ratio": float(shock.mean()),
        "shock_switch_within_1_step": within_k(shock, 1),
        "shock_switch_within_3_steps": within_k(shock, 3),
        "shock_switch_within_6_steps": within_k(shock, 6),
        "fast_transition_count": int(pd.to_numeric(df.get("fast_transition_flag", 0), errors="coerce").fillna(0).sum()),
        "exit_pressure_trigger_count": int(pd.to_numeric(df.get("exit_pressure_trigger_flag", 0), errors="coerce").fillna(0).sum()),
        "delayed_hold_count": int(pd.to_numeric(df.get("delayed_hold_flag", 0), errors="coerce").fillna(0).sum()),
        "candidate_stable_mismatch_ratio": float(mismatch.mean()),
        "mean_exit_pressure": float(pd.to_numeric(df.get("exit_pressure_current_state", 0), errors="coerce").fillna(0).mean()),
        "p95_exit_pressure": float(pd.to_numeric(df.get("exit_pressure_current_state", 0), errors="coerce").fillna(0).quantile(0.95)),
    }


def build_validation_report(df: pd.DataFrame, stats: pd.DataFrame, durations: pd.DataFrame, no_leakage: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    state_dist = df["stable_state"].value_counts(normalize=True).reindex(STATES).fillna(0).to_dict()
    churn = float(df["stable_switch_flag"].sum() / max(len(df), 1)) if "stable_switch_flag" in df else np.nan
    duration_summary = {}
    if not durations.empty:
        for s, g in durations.groupby("state"):
            duration_summary[s] = {
                "segments": int(len(g)),
                "mean_steps": float(g["n_steps"].mean()),
                "median_steps": float(g["n_steps"].median()),
                "short_1_step_ratio": float((g["n_steps"] <= 1).mean()),
            }
    report = {
        "rows": int(len(df)),
        "time_start": str(df["time"].min()),
        "time_end": str(df["time"].max()),
        "no_leakage": no_leakage,
        "state_distribution": {k: float(v) for k, v in state_dist.items()},
        "switch_rate_per_row": churn,
        "duration_summary": duration_summary,
        "scenario_review": scenario_review(df, model_config),
        "responsiveness_diagnostics": responsiveness_diagnostics(df, model_config),
        "segment_quality_summary": model_config.get("_segment_quality_summary", {}),
        "semantic_profile_brief": {
            row["state"]: {
                "count": int(row["count"]),
                "prop": float(row["proportion"]),
                "cascade_mean": float(row.get("e_cascade_mean", np.nan)),
                "rejection_mean": float(row.get("e_rejection_mean", np.nan)),
                "active_mean": float(row.get("e_active_mean", np.nan)),
                "quiet_mean": float(row.get("e_quiet_mean", np.nan)),
                "low_vol_mean": float(row.get("e_low_vol_mean", np.nan)),
                "range_compression_mean": float(row.get("e_range_compression_mean", np.nan)),
                "conflict_mean": float(row.get("e_conflict_mean", np.nan)),
            }
            for _, row in stats.iterrows()
        },
    }
    return report


def write_validation_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = []
    lines.append("# Six-State Engine Validation Report")
    lines.append("")
    lines.append(f"- Rows: {report['rows']}")
    lines.append(f"- Time range: {report['time_start']} → {report['time_end']}")
    lines.append(f"- No-leakage passed: {report.get('no_leakage', {}).get('passed')}")
    lines.append(f"- Switch rate per row: {report.get('switch_rate_per_row'):.6f}")
    lines.append("")
    lines.append("## State Distribution")
    for s, v in report["state_distribution"].items():
        lines.append(f"- {s}: {v:.4%}")
    lines.append("")
    lines.append("## Duration Summary")
    for s, d in report["duration_summary"].items():
        lines.append(f"- {s}: segments={d['segments']}, mean_steps={d['mean_steps']:.2f}, median_steps={d['median_steps']:.2f}, short_1_step_ratio={d['short_1_step_ratio']:.2%}")
    lines.append("")
    lines.append("## Responsiveness Diagnostics")
    rd = report.get("responsiveness_diagnostics", {})
    for k, v in rd.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.6f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    sq = report.get("segment_quality_summary", {}) or {}
    lines.append("## Segment Quality Summary")
    if not sq:
        lines.append("- No segment quality summary available.")
    else:
        lines.append(f"- Segments: {sq.get('segments')}")
        for k, v in (sq.get('issue_counts') or {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("### By State")
        for s, d in (sq.get('by_state') or {}).items():
            lines.append(f"- {s}: segments={d.get('segments')}, median_abs_return_bps={d.get('median_abs_return_bps'):.2f}, median_range_bps={d.get('median_range_bps'):.2f}, mean_trading_quality={d.get('mean_trading_trigger_quality'):.3f}, weak_segment_ratio={d.get('weak_segment_ratio'):.2%}")
    lines.append("")
    lines.append("## Scenario Review")
    for name, d in report["scenario_review"].items():
        lines.append(f"### {name}")
        lines.append(f"- Note: {d.get('note')}")
        if not d.get("available"):
            lines.append(f"- {d.get('message')}")
            continue
        lines.append(f"- Rows: {d['rows']}")
        lines.append(f"- Primary state: {d['primary_state']}")
        lines.append(f"- Expected primary: {d.get('expected_primary')}")
        lines.append(f"- Pass expectation: {d.get('pass_expectation')}")
        lines.append(f"- State distribution: {d.get('state_distribution')}")
        lines.append(f"- Top path context 6h: {d.get('path_context_top_6h')}")
        lines.append(f"- Top path label 6h: {d.get('path_label_top_6h')}")
        lines.append(f"- Top path context 24h: {d.get('path_context_top_24h')}")
        lines.append(f"- Top path label 24h: {d.get('path_label_top_24h')}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_all_reports(output_dir: str | Path, df: pd.DataFrame, no_leakage_report: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    defs = state_definitions_frame()
    stats = compute_state_feature_stats(df)
    durations = compute_duration_segments(df)
    samples = typical_scenarios(df)
    transitions = transition_events(df)
    segment_summary = save_segment_quality_outputs(output_dir, df, model_config)
    model_config_for_report = dict(model_config)
    model_config_for_report["_segment_quality_summary"] = segment_summary
    report = build_validation_report(df, stats, durations, no_leakage_report, model_config_for_report)

    defs.to_csv(output_dir / "state_definitions.csv", index=False)
    stats.to_csv(output_dir / "state_feature_stats.csv", index=False)
    durations.to_csv(output_dir / "state_duration_segments.csv", index=False)
    samples.to_csv(output_dir / "typical_scenarios.csv", index=False)
    transitions.to_csv(output_dir / "transition_events.csv", index=False)
    write_json(output_dir / "validation_report.json", report)
    write_validation_markdown(output_dir / "validation_report.md", report)
    return report
