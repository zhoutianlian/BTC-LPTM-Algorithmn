from __future__ import annotations

STATES = ["ST", "VT", "RC", "RHA", "HPEM", "AMB"]
STATE_ID = {s: i for i, s in enumerate(STATES)}

STATE_DEFINITIONS = {
    "ST": {"id": 0, "name": "Stable Trend", "cn": "稳定趋势", "meaning": "有序趋势；清算或主动趋势有一定方向，但不是级联或高吸收拒绝。"},
    "VT": {"id": 1, "name": "Volatile Trend", "cn": "波动趋势", "meaning": "主动交易主导的高波动趋势；PLIE 不是主解释变量。"},
    "RC": {"id": 2, "name": "Range-bound / Consolidation", "cn": "区间震荡或整理", "meaning": "低清算压力、低主动运动、低路径解释力。"},
    "RHA": {"id": 3, "name": "Reversal / High Absorption", "cn": "反转或高吸收", "meaning": "清算压力仍在，但价格路径拒绝该压力，forced flow 被吸收或接管。"},
    "HPEM": {"id": 4, "name": "High Pressure / Extreme Move", "cn": "高压力极端运动", "meaning": "清算压力穿透并被价格路径放大，可能发生 squeeze / flush / cascade。"},
    "AMB": {"id": 5, "name": "Ambiguous / No-trade", "cn": "模糊或观望", "meaning": "证据冲突、混合压力、低质量或不适合高置信判断。"},
}

DIRECTIONAL_CONTEXTS = {"path_directional_core", "path_directional_weak"}
NEUTRAL_CONTEXTS = {"path_neutral_pressure"}
MIXED_CONTEXTS = {"path_mixed_pressure"}

CASCADE_LABELS = {"path_cascade_transmission"}
BASELINE_LABELS = {"path_baseline_transmission", "path_partial_absorption"}
REJECTION_LABELS = {"path_pressure_rejection", "path_reversal_takeover", "path_full_absorption_stall"}
ACTIVE_LABELS = {
    "path_active_dominance_up", "path_active_dominance_down",
    "path_normal_active_dominance_up", "path_normal_active_dominance_down",
    "path_mixed_active_breakout_up", "path_mixed_active_breakout_down",
    "path_normal_mixed_active_breakout_up", "path_normal_mixed_active_breakout_down",
}
QUIET_LABELS = {"path_quiet_no_pressure"}
MIXED_CHOP_LABELS = {"path_mixed_pressure_chop"}
