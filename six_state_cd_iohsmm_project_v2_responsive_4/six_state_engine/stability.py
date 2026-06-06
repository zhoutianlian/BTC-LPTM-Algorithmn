from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from .constants import STATES
from .evidence import EvidenceBuilder


class ProductionStabilityLayer:
    """Production stable-state layer with responsiveness and V2.5 price-expectation exit pressure.

    The layer keeps the six-state semantic definitions intact. It converts filtered belief to
    stable_state with EWMA / hysteresis / minimum duration, while allowing well-formed causal
    evidence to reduce old-state inertia:

    - V2.2 state invalidity / price-shock fast lane.
    - V2.5 price-behavior expectation: if the causal price path since stable-state entry no
      longer matches HPEM/RHA/RC/VT/ST expectations, targeted exit pressure can be added.
    """

    def __init__(self, model_config: dict[str, Any]):
        self.cfg = model_config
        self.states = model_config.get("model", {}).get("states", STATES)
        self.st_cfg = model_config.get("stability", {})
        self.lam = float(self.st_cfg.get("ewma_lambda", 0.75))
        self.belief_gap_min = float(self.st_cfg.get("belief_gap_min", 0.08))
        self.tau_in = self.st_cfg.get("tau_in", {})
        self.tau_out = self.st_cfg.get("tau_out", {})
        self.min_duration = self.st_cfg.get("min_duration", {})

        resp = model_config.get("responsiveness", {}) if isinstance(model_config.get("responsiveness", {}), dict) else {}
        self.resp_enabled = bool(resp.get("enabled", True))
        self.price_shock_cfg = resp.get("price_shock", {}) if isinstance(resp.get("price_shock", {}), dict) else {}
        self.exit_cfg = resp.get("exit_pressure", {}) if isinstance(resp.get("exit_pressure", {}), dict) else {}
        self.exit_decay = float(self.exit_cfg.get("decay", 0.55))
        self.exit_soft = float(self.exit_cfg.get("soft_threshold", 0.45))
        self.exit_hard = float(self.exit_cfg.get("hard_threshold", 0.65))
        self.exit_lambda = float(self.exit_cfg.get("exit_lambda", 0.58))
        self.exit_tau_discount = self.exit_cfg.get("tau_discount", {})
        self.exit_gap_discount = float(self.exit_cfg.get("gap_discount", 0.03))
        self.hard_exit_extra_discount = float(self.exit_cfg.get("hard_exit_extra_discount", 0.04))
        self.exit_min_duration = self.exit_cfg.get("min_duration_for_exit_discount", {})

        self.shock_lambda = float(self.price_shock_cfg.get("shock_lambda", 0.55))
        self.shock_tau_discount = self.price_shock_cfg.get("tau_discount", {})
        self.shock_gap_discount = float(self.price_shock_cfg.get("gap_discount", 0.03))
        self.local_cooldown_steps = int(self.price_shock_cfg.get("local_cooldown_steps", 1))
        self.shock_allowed = self.price_shock_cfg.get("allowed", {})

        # V2.5: price-behavior expectation can add targeted exit pressure.
        pe_cfg = model_config.get("state_price_expectation", {}) if isinstance(model_config.get("state_price_expectation", {}), dict) else {}
        self.pe_cfg = pe_cfg
        self.pe_enabled = bool(pe_cfg.get("enabled", False)) and bool(pe_cfg.get("use_in_stability", True))
        self.pe_state_cfg = pe_cfg.get("states", {}) if isinstance(pe_cfg.get("states", {}), dict) else {}
        pe_boost = pe_cfg.get("stability_boost", {}) if isinstance(pe_cfg.get("stability_boost", {}), dict) else {}
        self.pe_soft = float(pe_boost.get("soft_threshold", 0.55))
        self.pe_hard = float(pe_boost.get("hard_threshold", 0.75))
        self.pe_lambda = float(pe_boost.get("lambda", 0.52))
        self.pe_gap_discount = float(pe_boost.get("gap_discount", 0.02))
        self.pe_tau_discount = pe_boost.get("tau_discount", {}) if isinstance(pe_boost.get("tau_discount", {}), dict) else {}
        self.pe_hard_extra_discount = float(pe_boost.get("hard_exit_extra_discount", 0.05))
        self.pe_override_candidate = bool(pe_boost.get("allow_target_override", True))
        self.pe_target_min_belief = float(pe_boost.get("target_min_stable_belief", 0.12))
        self.pe_hard_target_min_belief = float(pe_boost.get("hard_target_min_stable_belief", 0.10))
        self.pe_target_belief_boost = float(pe_boost.get("target_belief_boost", 0.06))

    @staticmethod
    def _f(row: pd.Series, key: str, default: float = 0.0) -> float:
        try:
            x = float(row.get(key, default))
            if not np.isfinite(x):
                return float(default)
            return x
        except Exception:
            return float(default)

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _discount(self, mapping: dict[str, Any], state: str, default: float = 0.0) -> float:
        try:
            return float(mapping.get(state, default))
        except Exception:
            return float(default)

    def _pressure_direction(self, row: pd.Series) -> int:
        for key in ("plie_direction", "state_pressure_direction"):
            val = self._f(row, key, 0.0)
            if val > 0:
                return 1
            if val < 0:
                return -1
        hs = int(round(self._f(row, "hmm_state", 0)))
        if hs in (1, 2):
            return 1
        if hs in (4, 5):
            return -1
        return 0

    def _gate(self, row: pd.Series, target: str) -> tuple[bool, str]:
        if target == "AMB":
            ok = (
                self._f(row, "e_conflict") > 0.40
                or self._f(row, "e_quality_bad") > 0.38
                or (self._f(row, "e_cross_window_conflict") > 0.65 and max(self._f(row, "e_rha_strict"), self._f(row, "e_cascade"), self._f(row, "e_active"), self._f(row, "e_quiet")) < 0.45)
                or (self._f(row, "e_mixed_context") > 0.55 and self._f(row, "e_quiet") < 0.45 and self._f(row, "e_active") < 0.45)
            )
            return bool(ok), "AMB gate: conflict/data-quality/cross-window uncertainty"
        if target == "RHA":
            ok = self._f(row, "e_directional_pressure") > 0.25 and (self._f(row, "e_rha_strict") > 0.28 or self._f(row, "e_rejection") > 0.42) and self._f(row, "e_neutral_context") < 0.60
            return bool(ok), "RHA gate: directional pressure + strict rejection/takeover/stall"
        if target == "HPEM":
            ok = self._f(row, "e_directional_pressure") > 0.25 and (self._f(row, "e_cascade") > 0.35 or self._f(row, "e_pressure_strength") > 0.62 or self._f(row, "e_strong_entry") > 0.5) and self._f(row, "e_neutral_context") < 0.70
            return bool(ok), "HPEM gate: directional pressure + cascade/strong pressure"
        if target == "VT":
            ok = self._f(row, "e_active") > 0.32 and (self._f(row, "e_volatility") > 0.25 or self._f(row, "e_trend_strength") > 0.30 or self._f(row, "e_neutral_or_mixed") > 0.50) and not (self._f(row, "e_directional_pressure") > 0.65 and self._f(row, "e_cascade") > 0.55)
            return bool(ok), "VT gate: active dominance + price trend/volatility with low liquidation explanation"
        if target == "RC":
            ok = ((self._f(row, "e_quiet") > 0.30 or (self._f(row, "e_neutral_context") > 0.50 and self._f(row, "e_low_vol") > 0.55 and self._f(row, "e_range_compression") > 0.45 and self._f(row, "e_active") < 0.45)) and self._f(row, "e_conflict") < 0.48 and self._f(row, "e_quality_bad") < 0.55 and self._f(row, "e_pressure_strength") < 0.72)
            return bool(ok), "RC gate: neutral/quiet + low vol/trend + low conflict"
        if target == "ST":
            ok = self._f(row, "e_baseline") > 0.18 and self._f(row, "e_conflict") < 0.55 and self._f(row, "e_cascade") < 0.70
            return bool(ok), "ST gate: baseline/orderly transmission and low conflict"
        return True, "default gate"

    def _state_invalidity(self, row: pd.Series, state: str) -> tuple[float, str]:
        f = self._f
        if state == "HPEM":
            val = 0.35 * f(row, "e_shock_against_plie") * max(f(row, "e_rejection"), f(row, "e_absorption")) + 0.25 * f(row, "e_rejection") + 0.20 * f(row, "e_absorption") + 0.20 * (1 - f(row, "e_cascade"))
            return self._clip01(val), "HPEM invalidity: reverse shock/rejection/absorption or cascade decay"
        if state == "RHA":
            val = 0.30 * f(row, "e_neutral_context") + 0.25 * f(row, "e_active") * (1 - f(row, "e_directional_pressure")) + 0.20 * f(row, "e_quiet") + 0.15 * (1 - f(row, "e_rha_strict")) + 0.10 * f(row, "e_shock_aligned_plie") * f(row, "e_cascade")
            return self._clip01(val), "RHA invalidity: pressure decays, active/quiet takeover, or rejection fades"
        if state == "RC":
            val = 0.35 * f(row, "e_short_impulse_1h") + 0.25 * f(row, "e_active") + 0.20 * f(row, "e_pressure_strength") + 0.10 * f(row, "e_cascade") + 0.10 * (1 - f(row, "e_range_compression"))
            return self._clip01(val), "RC invalidity: short impulse, active breakout, pressure or compression loss"
        if state == "VT":
            val = 0.30 * f(row, "e_quiet") + 0.20 * (1 - f(row, "e_trend_strength")) + 0.20 * (1 - f(row, "e_volatility")) + 0.20 * f(row, "e_directional_pressure") * f(row, "e_cascade") + 0.10 * f(row, "e_conflict")
            return self._clip01(val), "VT invalidity: trend/vol decay, quiet compression, or liquidation cascade emerges"
        if state == "ST":
            val = 0.25 * f(row, "e_cascade") + 0.25 * f(row, "e_rha_strict") + 0.20 * f(row, "e_active") * f(row, "e_volatility") + 0.15 * f(row, "e_quiet") + 0.15 * f(row, "e_conflict")
            return self._clip01(val), "ST invalidity: cascade, strict RHA, active-vol breakout, quiet or conflict"
        if state == "AMB":
            val = 1 - max(f(row, "e_conflict"), f(row, "e_quality_bad"), f(row, "e_cross_window_conflict"))
            return self._clip01(val), "AMB invalidity: clarity recovered"
        return 0.0, "no invalidity rule"

    def _shock_candidate_allowed(self, row: pd.Series, old: str, target: str, cooldown_remaining: int) -> tuple[bool, str]:
        if not self.resp_enabled or not bool(self.price_shock_cfg.get("enabled", True)):
            return False, "shock disabled"
        if cooldown_remaining > 0:
            return False, f"shock cooldown={cooldown_remaining}"
        if target == old:
            return False, "same state"
        allowed_targets = self.shock_allowed.get(old, [])
        if allowed_targets and target not in allowed_targets:
            return False, f"target {target} not in fast-lane allowed list for {old}"
        min_shock = float(self.price_shock_cfg.get("min_price_shock_1h", 0.72))
        min_impulse = float(self.price_shock_cfg.get("min_short_impulse_1h", 0.45))
        min_trend = float(self.price_shock_cfg.get("min_trend_strength_1h", 0.45))
        if self._f(row, "e_shock_data_valid") < 0.5:
            return False, "shock data invalid"
        if self._f(row, "e_price_shock_1h") < min_shock or self._f(row, "e_short_impulse_1h") < min_impulse:
            return False, "shock below threshold"
        if self._f(row, "trend_strength_1h") < min_trend and self._f(row, "e_trend_strength") < min_trend:
            return False, "trend strength below threshold"
        if target == "HPEM":
            ok = self._f(row, "e_shock_aligned_plie") > 0.15 and self._f(row, "e_directional_pressure") > 0.20 and (self._f(row, "e_cascade") > 0.25 or self._f(row, "e_pressure_strength") > 0.45)
            return bool(ok), "fast HPEM: 1h shock aligned with directional PLIE/cascade"
        if target == "RHA":
            ok = self._f(row, "e_shock_against_plie") > 0.15 and max(self._f(row, "e_rha_strict"), self._f(row, "e_rejection"), self._f(row, "e_absorption")) > 0.25
            return bool(ok), "fast RHA: 1h shock against PLIE with absorption/rejection"
        if target == "VT":
            ok = self._f(row, "e_shock_active_breakout") > 0.12 and (self._f(row, "e_active") > 0.25 or self._f(row, "e_neutral_or_mixed") > 0.45)
            return bool(ok), "fast VT: neutral/mixed active price shock"
        if target == "AMB":
            ok = self._f(row, "e_conflict") > 0.40 or self._f(row, "e_quality_bad") > 0.35
            return bool(ok), "fast AMB: shock plus conflict/data risk"
        return False, f"no shock fast lane for {target}"

    def _price_impact_score(self, row: pd.Series, abs_ret: float, rng: float) -> float:
        return self._clip01(max(
            min(abs_ret / 120.0, 1.0), min(rng / 220.0, 1.0),
            min(self._f(row, "realized_vol_1h_bps") / 100.0, 1.0),
            min(self._f(row, "jump_proxy_1h") / 0.75, 1.0),
            min(self._f(row, "max_jump_z_1h") / 5.0, 1.0),
            self._f(row, "e_short_impulse_1h"),
        ))

    def _select_price_expectation_target(self, row: pd.Series, state: str, preferred: str, stable_b_prev: np.ndarray, hard: bool = False) -> str:
        candidates: list[str] = []
        if preferred and preferred in self.states and preferred != state:
            candidates.append(preferred)
        for t in {"HPEM": ["RHA", "VT", "ST", "AMB"], "RHA": ["HPEM", "ST", "RC", "AMB"], "RC": ["VT", "ST", "HPEM", "AMB"], "VT": ["RC", "ST", "AMB", "HPEM"], "ST": ["RC", "VT", "HPEM", "AMB"]}.get(state, ["AMB"]):
            if t not in candidates and t != state:
                candidates.append(t)
        min_b = self.pe_hard_target_min_belief if hard else self.pe_target_min_belief
        best, best_b = "", -1.0
        for target in candidates:
            ok, _ = self._gate(row, target)
            if not ok:
                continue
            try:
                b = float(stable_b_prev[self.states.index(target)])
            except Exception:
                b = 0.0
            if b >= min_b and b > best_b:
                best, best_b = target, b
        return best

    def _price_expectation_signal(self, row: pd.Series, state: str, duration: int, entry_price: float, running_high: float, running_low: float, pressure_ref: int, hpem_mfe_prev: float, rha_mfe_prev: float) -> dict[str, Any]:
        if not self.pe_enabled:
            return {"score": 1.0, "invalidity": 0.0, "flag": 0, "issue": "", "target": "", "reason": "disabled", "hpem_mfe": hpem_mfe_prev, "rha_mfe": rha_mfe_prev}
        price = self._f(row, "price", np.nan)
        if not np.isfinite(price) or not np.isfinite(entry_price) or entry_price <= 0:
            return {"score": 1.0, "invalidity": 0.0, "flag": 0, "issue": "", "target": "", "reason": "no price", "hpem_mfe": hpem_mfe_prev, "rha_mfe": rha_mfe_prev}
        ret = 10000.0 * np.log(price / entry_price)
        rng = 0.0
        if np.isfinite(running_high) and np.isfinite(running_low) and running_low > 0:
            rng = 10000.0 * np.log(max(running_high, price) / min(running_low, price))
        abs_ret = abs(ret)
        cfg = self.pe_state_cfg.get(state, {}) if isinstance(self.pe_state_cfg.get(state, {}), dict) else {}
        impact = self._price_impact_score(row, abs_ret, rng)
        score, invalidity, flag, issue, target, reason = 1.0, 0.0, 0, "", "", "OK"
        hpem_mfe, rha_mfe = hpem_mfe_prev, rha_mfe_prev
        if state == "HPEM":
            aligned = pressure_ref * ret if pressure_ref != 0 else 0.0
            hpem_mfe = max(float(hpem_mfe_prev), aligned)
            giveback = max(0.0, hpem_mfe - aligned)
            giveback_ratio = giveback / max(abs(hpem_mfe), 1e-6)
            no_response = duration >= int(cfg.get("grace_bars", 2)) and hpem_mfe < float(cfg.get("min_aligned_mfe_bps", 80)) and impact < float(cfg.get("min_price_impact", 0.45)) and self._f(row, "realized_vol_1h_bps") < float(cfg.get("min_realized_vol_1h_bps", 60)) and self._f(row, "jump_proxy_1h") < float(cfg.get("min_jump_proxy_1h", 0.35))
            decay = hpem_mfe >= float(cfg.get("min_aligned_mfe_bps", 80)) and giveback >= float(cfg.get("giveback_abs_bps", 120)) and giveback_ratio >= float(cfg.get("giveback_ratio", 0.55))
            adverse = aligned <= -float(cfg.get("adverse_move_bps", 60))
            if adverse or decay or no_response:
                flag = 1; issue = "HPEM_ADVERSE_PRICE" if adverse else "HPEM_GIVEBACK_DECAY" if decay else "HPEM_NO_PRICE_RESPONSE"
                invalidity = 0.90 if adverse else 0.82 if decay else 0.68
                target = "RHA" if max(self._f(row, "e_rejection"), self._f(row, "e_absorption"), self._f(row, "e_rha_strict")) > 0.25 else "VT" if self._f(row, "e_active") > 0.35 and self._f(row, "e_neutral_or_mixed") > 0.35 else "AMB" if self._f(row, "e_conflict") > 0.45 else "ST"
                reason = f"{issue}: aligned={aligned:.1f}, mfe={hpem_mfe:.1f}, giveback={giveback:.1f}, impact={impact:.2f}"
        elif state == "RHA":
            reversal_move = (-pressure_ref) * ret if pressure_ref != 0 else 0.0
            rha_mfe = max(float(rha_mfe_prev), reversal_move)
            follow_plie = pressure_ref * ret if pressure_ref != 0 else 0.0
            invalid_follow = follow_plie >= float(cfg.get("invalid_if_price_follows_plie_bps", 80))
            weak_no_reversal = duration >= int(cfg.get("weak_no_reversal_confirm_bars", 4)) and rha_mfe < float(cfg.get("reversal_min_mfe_bps", 80)) and bool(cfg.get("exit_weak_no_reversal", True))
            stall = duration >= int(cfg.get("stall_confirm_bars", 4)) and abs_ret <= float(cfg.get("stall_max_abs_move_bps", 60)) and max(self._f(row, "e_absorption"), self._f(row, "e_rejection")) >= float(cfg.get("stall_requires_absorption", 0.55)) and bool(cfg.get("exit_on_stall", False))
            if invalid_follow or weak_no_reversal or stall:
                flag = 1; issue = "RHA_INVALID_FOLLOWS_PLIE" if invalid_follow else "RHA_WEAK_NO_REVERSAL" if weak_no_reversal else "RHA_STALL_NOT_DIRECTIONAL"
                invalidity = 0.88 if invalid_follow else 0.62 if weak_no_reversal else 0.55
                target = "HPEM" if invalid_follow and self._f(row, "e_cascade") > 0.35 else "ST" if invalid_follow else "RC"
                reason = f"{issue}: reversal_mfe={rha_mfe:.1f}, follow_plie={follow_plie:.1f}"
        elif state == "RC":
            drift = abs_ret > float(cfg.get("max_entry_to_current_abs_return_bps", 80))
            range_break = rng > float(cfg.get("max_entry_to_current_range_bps", 140))
            trending = self._f(row, "trend_strength_6h") > float(cfg.get("max_trend_strength_6h", 0.50)) and self._f(row, "trend_consistency_6h") > float(cfg.get("max_trend_consistency_6h", 0.58))
            if duration >= int(cfg.get("exit_confirm_bars", 2)) and (drift or range_break or trending):
                flag = 1; issue = "RC_PRICE_TOO_LARGE_OR_TRENDING"; invalidity = 0.72 if (range_break or trending) else 0.62
                target = "VT" if self._f(row, "e_active") > 0.35 or self._f(row, "e_volatility") > 0.45 else "ST" if self._f(row, "e_baseline") > 0.25 or self._f(row, "e_trend_strength") > 0.35 else "HPEM" if self._f(row, "e_cascade") > 0.35 else "VT"
                reason = f"{issue}: abs_ret={abs_ret:.1f}, range={rng:.1f}, trend6={self._f(row,'trend_strength_6h'):.2f}"
        elif state == "VT":
            low_move = abs_ret < float(cfg.get("min_entry_to_current_abs_return_bps", 60)) and rng < float(cfg.get("min_entry_to_current_range_bps", 120))
            weak_vol = self._f(row, "realized_vol_1h_bps") < float(cfg.get("min_realized_vol_1h_bps", 45)) and self._f(row, "realized_vol_6h_bps") < float(cfg.get("min_realized_vol_6h_bps", 100))
            weak_trend = self._f(row, "trend_strength_6h") < float(cfg.get("min_trend_strength_6h", 0.40))
            if duration >= int(cfg.get("low_move_confirm_bars", 2)) and (low_move and (weak_vol or weak_trend)):
                flag = 1; issue = "VT_LOW_MOVE_OR_LOW_VOL"; invalidity = 0.58
                target = "RC" if self._f(row, "e_quiet") > 0.35 or self._f(row, "e_range_compression") > 0.55 else "ST"
                reason = f"{issue}: abs_ret={abs_ret:.1f}, range={rng:.1f}, rv1={self._f(row,'realized_vol_1h_bps'):.1f}"
        elif state == "ST":
            flat = abs_ret < float(cfg.get("min_entry_to_current_abs_return_bps", 35)) and rng < float(cfg.get("min_entry_to_current_range_bps", 70))
            weak = self._f(row, "trend_strength_6h") < float(cfg.get("min_trend_strength_6h", 0.38)) or self._f(row, "trend_consistency_6h") < float(cfg.get("min_trend_consistency_6h", 0.50))
            extreme = abs_ret > float(cfg.get("max_entry_to_current_abs_return_bps", 220)) or rng > float(cfg.get("max_entry_to_current_range_bps", 420)) or self._f(row, "realized_vol_1h_bps") > float(cfg.get("max_realized_vol_1h_bps", 80))
            if duration >= int(cfg.get("exit_confirm_bars", 2)) and (flat or weak or extreme):
                flag = 1; issue = "ST_FLAT_NO_TREND" if flat else "ST_TOO_EXTREME" if extreme else "ST_WEAK_TREND"
                invalidity = 0.62 if flat else 0.68 if extreme else 0.50
                target = "RC" if flat or weak else "VT" if self._f(row, "e_active") > 0.35 or self._f(row, "e_neutral_or_mixed") > 0.45 else "HPEM" if self._f(row, "e_cascade") > 0.35 else "VT"
                reason = f"{issue}: abs_ret={abs_ret:.1f}, range={rng:.1f}, trend6={self._f(row,'trend_strength_6h'):.2f}"
        score = self._clip01(1.0 - invalidity)
        return {"score": score, "invalidity": invalidity, "flag": flag, "issue": issue, "target": target, "reason": reason, "hpem_mfe": hpem_mfe, "rha_mfe": rha_mfe}

    def _top_evidence(self, row: pd.Series, state: str) -> str:
        existing = row.get("top_evidence_features", "")
        if isinstance(existing, str) and existing:
            return existing
        return EvidenceBuilder.top_evidence_for_state(row, state)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        b_cols = [f"b_{s}" for s in self.states]
        b = out[b_cols].to_numpy(float)
        stable_b = np.zeros_like(b)
        stable_b[0] = b[0]

        lists: dict[str, list[Any]] = {k: [] for k in [
            "stable_state", "stable_state_duration", "stable_switch_flag", "from_state", "transition_reason", "top_evidence_features",
            "state_invalidity_score", "state_invalidity_reason", "exit_pressure_current_state", "fast_transition_flag",
            "exit_pressure_trigger_flag", "delayed_hold_flag", "effective_tau_in", "effective_belief_gap_min", "effective_ewma_lambda", "fast_transition_reason",
            "price_expectation_live_score", "price_expectation_live_issue", "price_expectation_live_exit_flag", "price_expectation_live_exit_target",
            "price_expectation_live_reason", "price_expectation_invalidity_score", "price_expectation_override_flag"
        ]}

        stable = str(out.iloc[0].get("candidate_state", self.states[int(np.nanargmax(b[0]))]))
        duration = 1; exit_pressure = 0.0; fast_cooldown = 0
        entry_price = self._f(out.iloc[0], "price", np.nan)
        running_high = entry_price; running_low = entry_price
        pressure_ref = self._pressure_direction(out.iloc[0])
        hpem_mfe = 0.0; rha_mfe = 0.0

        def append_row(**kwargs: Any) -> None:
            for k, v in kwargs.items():
                lists[k].append(v)

        append_row(stable_state=stable, stable_state_duration=duration, stable_switch_flag=1, from_state="",
                   transition_reason=f"initial_state={stable}; {self._top_evidence(out.iloc[0], stable)}",
                   top_evidence_features=self._top_evidence(out.iloc[0], stable), state_invalidity_score=0.0,
                   state_invalidity_reason="initial", exit_pressure_current_state=0.0, fast_transition_flag=0,
                   exit_pressure_trigger_flag=0, delayed_hold_flag=0, effective_tau_in=float(self.tau_in.get(stable, 0.5)),
                   effective_belief_gap_min=self.belief_gap_min, effective_ewma_lambda=self.lam, fast_transition_reason="initial",
                   price_expectation_live_score=1.0, price_expectation_live_issue="", price_expectation_live_exit_flag=0,
                   price_expectation_live_exit_target="", price_expectation_live_reason="initial", price_expectation_invalidity_score=0.0,
                   price_expectation_override_flag=0)

        for t in range(1, len(out)):
            row = out.iloc[t]
            raw_cand = str(row.get("candidate_state", self.states[int(np.nanargmax(b[t]))]))
            old = stable; old_i = self.states.index(stable)
            price_t = self._f(row, "price", entry_price)
            if np.isfinite(price_t) and price_t > 0:
                running_high = max(running_high, price_t) if np.isfinite(running_high) else price_t
                running_low = min(running_low, price_t) if np.isfinite(running_low) else price_t

            base_invalidity, base_reason = self._state_invalidity(row, stable)
            pe = self._price_expectation_signal(row, stable, duration, entry_price, running_high, running_low, pressure_ref, hpem_mfe, rha_mfe)
            hpem_mfe = float(pe.get("hpem_mfe", hpem_mfe)); rha_mfe = float(pe.get("rha_mfe", rha_mfe))
            pe_invalidity = float(pe.get("invalidity", 0.0)) if int(pe.get("flag", 0)) == 1 else 0.0
            invalidity = self._clip01(max(base_invalidity, pe_invalidity))
            invalidity_reason = base_reason if pe_invalidity <= 0 else f"{base_reason}; price_expectation={pe.get('issue','')}"
            exit_pressure = self.exit_decay * exit_pressure + (1 - self.exit_decay) * invalidity
            exit_min_d = int(self.exit_min_duration.get(stable, self.min_duration.get(stable, 1)))
            exit_soft_active = self.resp_enabled and bool(self.exit_cfg.get("enabled", True)) and duration >= exit_min_d and exit_pressure >= self.exit_soft
            exit_hard_active = self.resp_enabled and bool(self.exit_cfg.get("enabled", True)) and duration >= exit_min_d and exit_pressure >= self.exit_hard

            cand = raw_cand; pe_override_used = False
            if self.pe_override_candidate and int(pe.get("flag", 0)) == 1:
                chosen = self._select_price_expectation_target(row, stable, str(pe.get("target", "")), stable_b[t-1], hard=pe_invalidity >= self.pe_hard)
                if chosen and chosen != stable:
                    cand = chosen; pe_override_used = (raw_cand == stable)
            cand_i = self.states.index(cand)

            shock_allowed, shock_reason = self._shock_candidate_allowed(row, stable, cand, fast_cooldown)
            shock_active = bool(shock_allowed and cand != stable)

            lambda_eff = self.lam
            if shock_active: lambda_eff = min(lambda_eff, self.shock_lambda)
            if exit_soft_active: lambda_eff = min(lambda_eff, self.exit_lambda)
            if pe_invalidity >= self.pe_soft: lambda_eff = min(lambda_eff, self.pe_lambda)
            stable_b[t] = lambda_eff * stable_b[t - 1] + (1 - lambda_eff) * b[t]

            switched = False; exit_trigger_used = False; shock_trigger_used = False; delayed_hold = False
            gate_reason = ""; tau_eff = float(self.tau_in.get(cand, 0.5)); gap_eff = self.belief_gap_min
            if cand != stable:
                gap = float(stable_b[t, cand_i] - stable_b[t, old_i])
                min_d = int(self.min_duration.get(stable, 1))
                gate_ok, gate_reason = self._gate(row, cand)
                if shock_active:
                    tau_eff -= self._discount(self.shock_tau_discount, cand, 0.0); gap_eff -= self.shock_gap_discount
                if exit_soft_active:
                    tau_eff -= self._discount(self.exit_tau_discount, cand, 0.0); gap_eff -= self.exit_gap_discount
                if pe_invalidity >= self.pe_soft:
                    tau_eff -= self._discount(self.pe_tau_discount, cand, 0.0); gap_eff -= self.pe_gap_discount
                tau_eff = max(0.05, tau_eff); gap_eff = max(-0.08, gap_eff)
                cand_belief_eff = float(stable_b[t, cand_i])
                if pe_override_used: cand_belief_eff = max(cand_belief_eff, self.pe_target_belief_boost)
                normal_entry = cand_belief_eff >= tau_eff and gap >= gap_eff and duration >= min_d and gate_ok
                old_exited = (stable_b[t, old_i] <= float(self.tau_out.get(stable, 0.35))) or exit_hard_active or (pe_invalidity >= self.pe_hard)
                exit_entry = old_exited and cand_belief_eff >= max(0.05, tau_eff - self.hard_exit_extra_discount - self.pe_hard_extra_discount) and duration >= min_d and gate_ok
                shock_entry = shock_active and cand_belief_eff >= max(0.05, tau_eff - 0.02) and duration >= min_d and gate_ok
                if normal_entry or exit_entry or shock_entry:
                    stable = cand; duration = 1; switched = True
                    exit_trigger_used = bool(exit_entry and not normal_entry); shock_trigger_used = bool(shock_entry and not normal_entry)
                    bits = []
                    if normal_entry: bits.append("normal_entry")
                    if exit_entry: bits.append("exit_pressure")
                    if shock_entry: bits.append("price_shock_fast_lane")
                    if pe_override_used: bits.append("price_expectation_override")
                    reason = (f"{old}->{cand}; raw_candidate={raw_cand}; triggers={'+'.join(bits)}; prob={self._f(row,'candidate_transition_probability'):.3f}; "
                              f"stable_belief={stable_b[t, cand_i]:.3f}; old_belief={stable_b[t, old_i]:.3f}; gap={gap:.3f}; tau_eff={tau_eff:.3f}; "
                              f"gap_eff={gap_eff:.3f}; lambda_eff={lambda_eff:.3f}; invalidity={invalidity:.3f}; exit_pressure={exit_pressure:.3f}; "
                              f"price_expectation={pe_invalidity:.3f}:{pe.get('reason','')}; {gate_reason}; {shock_reason}; {self._top_evidence(row, cand)}")
                    exit_pressure = 0.0
                    entry_price = price_t if np.isfinite(price_t) and price_t > 0 else entry_price
                    running_high = entry_price; running_low = entry_price
                    pressure_ref = self._pressure_direction(row)
                    hpem_mfe = 0.0; rha_mfe = 0.0
                    fast_cooldown = self.local_cooldown_steps if shock_trigger_used else max(0, fast_cooldown - 1)
                else:
                    duration += 1
                    delayed_hold = bool((exit_soft_active or shock_active or pe_invalidity >= self.pe_soft) and gate_ok)
                    reason = (f"hold={stable}; candidate={cand}; raw_candidate={raw_cand}; candidate_belief={stable_b[t, cand_i]:.3f}; "
                              f"old_belief={stable_b[t, old_i]:.3f}; gap={gap:.3f}; tau_eff={tau_eff:.3f}; gap_eff={gap_eff:.3f}; lambda_eff={lambda_eff:.3f}; "
                              f"gate_ok={gate_ok}; exit_pressure={exit_pressure:.3f}; invalidity={invalidity:.3f}; price_expectation={pe_invalidity:.3f}:{pe.get('issue','')}; "
                              f"shock_active={shock_active}; pe_override={pe_override_used}; {gate_reason}; {shock_reason}")
                    fast_cooldown = max(0, fast_cooldown - 1)
            else:
                duration += 1
                tau_eff = float(self.tau_in.get(stable, 0.5)); gap_eff = self.belief_gap_min
                reason = f"stay={stable}; lambda_eff={lambda_eff:.3f}; invalidity={invalidity:.3f}; exit_pressure={exit_pressure:.3f}; price_expectation={pe_invalidity:.3f}:{pe.get('issue','')}; {self._top_evidence(row, stable)}"
                fast_cooldown = max(0, fast_cooldown - 1)

            append_row(stable_state=stable, stable_state_duration=duration, stable_switch_flag=1 if switched else 0, from_state=old if switched else "", transition_reason=reason,
                       top_evidence_features=self._top_evidence(row, stable), state_invalidity_score=float(invalidity), state_invalidity_reason=invalidity_reason,
                       exit_pressure_current_state=float(exit_pressure), fast_transition_flag=1 if shock_trigger_used else 0, exit_pressure_trigger_flag=1 if exit_trigger_used else 0,
                       delayed_hold_flag=1 if delayed_hold else 0, effective_tau_in=float(tau_eff), effective_belief_gap_min=float(gap_eff), effective_ewma_lambda=float(lambda_eff),
                       fast_transition_reason=shock_reason, price_expectation_live_score=float(pe.get("score", 1.0)), price_expectation_live_issue=str(pe.get("issue", "")),
                       price_expectation_live_exit_flag=int(pe.get("flag", 0)), price_expectation_live_exit_target=str(pe.get("target", "")), price_expectation_live_reason=str(pe.get("reason", "")),
                       price_expectation_invalidity_score=float(pe_invalidity), price_expectation_override_flag=1 if pe_override_used else 0)

        for i, state in enumerate(self.states):
            out[f"stable_b_{state}"] = stable_b[:, i]
        out["stable_state_id"] = [self.states.index(s) for s in lists["stable_state"]]
        for k, v in lists.items():
            out[k] = v
        return out
