from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from .constants import STATES
from .utils import sigmoid


class CDIOHSMMFilter:
    def __init__(self, model_config: dict[str, Any]):
        self.cfg = model_config
        self.states = model_config.get("model", {}).get("states", STATES)
        self.k = len(self.states)
        self.d_max = int(model_config.get("model", {}).get("d_max", 96))
        self.emission_floor = float(model_config.get("model", {}).get("emission_floor", 1e-6))
        self.emission_power = float(model_config.get("model", {}).get("emission_power", 1.0))
        filt = model_config.get("filter", {})
        self.base_exit = filt.get("base_exit_logit", {})
        self.age_coef = filt.get("age_coef", {})
        self.hazard_clip = filt.get("hazard_clip", [0.01, 0.75])
        resp = model_config.get("responsiveness", {}) if isinstance(model_config.get("responsiveness", {}), dict) else {}
        self.resp_enabled = bool(resp.get("enabled", True))
        self.shock_filter_weights = resp.get("filter_shock_weights", {})
        self._ages = np.arange(1, self.d_max + 1, dtype=float)
        self._log_age = np.log1p(self._ages)

    @staticmethod
    def _get(e: dict[str, float], key: str, default: float = 0.0) -> float:
        return float(e.get(key, default))

    def _w(self, group: str, key: str, default: float) -> float:
        try:
            return float(self.shock_filter_weights.get(group, {}).get(key, default))
        except Exception:
            return float(default)

    def _state_extra_logit(self, state: str, e: dict[str, float], delta: dict[str, float]) -> float:
        if state == "HPEM":
            return (
                1.10 * self._get(e, "e_rejection")
                + 0.80 * self._get(e, "e_absorption")
                - 0.85 * self._get(e, "e_cascade")
                + 0.55 * max(self._get(delta, "e_rejection"), 0)
                + 0.35 * max(self._get(delta, "e_absorption"), 0)
                + self._w("exit", "HPEM_reverse_shock", 1.15) * self._get(e, "e_shock_against_plie") * max(self._get(e, "e_rejection"), self._get(e, "e_absorption"))
            )
        if state == "RHA":
            # RHA should exit when directional pressure decays, neutral/active dominance rises,
            # or strict rejection evidence fades.
            return (
                0.75 * self._get(e, "e_active")
                + 0.70 * self._get(e, "e_neutral_context")
                + 0.55 * (1 - self._get(e, "e_directional_pressure"))
                + 0.35 * (1 - self._get(e, "e_rha_strict"))
                + 0.25 * self._get(e, "e_quality_bad")
                + self._w("exit", "RHA_active_breakout", 0.85) * self._get(e, "e_shock_active_breakout")
                + self._w("exit", "RHA_rebreak_cascade", 0.65) * self._get(e, "e_shock_aligned_plie") * self._get(e, "e_cascade")
            )
        if state == "VT":
            return (
                0.55 * (1 - self._get(e, "e_active"))
                + 0.35 * (1 - max(self._get(e, "e_volatility"), self._get(e, "e_trend_strength")))
                + 0.35 * self._get(e, "e_conflict")
                + 0.25 * self._get(e, "e_directional_pressure") * self._get(e, "e_cascade")
                + self._w("exit", "VT_liq_cascade_shock", 0.75) * self._get(e, "e_shock_aligned_plie") * self._get(e, "e_directional_pressure") * self._get(e, "e_cascade")
            )
        if state == "RC":
            return (
                0.90 * self._get(e, "e_active")
                + 0.80 * self._get(e, "e_pressure_strength")
                + 0.45 * self._get(e, "e_volatility")
                + 0.35 * self._get(e, "e_jump")
                + 0.35 * self._get(e, "e_conflict")
                + self._w("exit", "RC_short_impulse", 1.00) * self._get(e, "e_short_impulse_1h")
            )
        if state == "ST":
            return (
                0.75 * self._get(e, "e_cascade")
                + 0.70 * self._get(e, "e_rejection")
                + 0.60 * self._get(e, "e_conflict")
                + self._w("exit", "ST_shock", 0.65) * max(self._get(e, "e_shock_aligned_plie"), self._get(e, "e_shock_against_plie"), self._get(e, "e_shock_active_breakout"))
            )
        if state == "AMB":
            clarity = 1 - max(self._get(e, "e_conflict"), self._get(e, "e_quality_bad"))
            return 1.25 * clarity
        return 0.0

    def _hazard_vec(self, state: str, e: dict[str, float], delta: dict[str, float]) -> np.ndarray:
        x = (
            float(self.base_exit.get(state, -2.2))
            + float(self.age_coef.get(state, 0.15)) * self._log_age
            + self._state_extra_logit(state, e, delta)
        )
        lo, hi = self.hazard_clip
        return np.clip(sigmoid(x), lo, hi)

    def _target_probs(self, prev_state: str, e: dict[str, float], delta: dict[str, float]) -> np.ndarray:
        ep = self._get
        logits = {
            "ST": (
                1.00 * ep(e, "e_baseline")
                + 0.55 * ep(e, "e_directional_pressure")
                + 0.45 * ep(e, "e_orderly_trend")
                + 0.35 * ep(e, "e_moderate_volatility")
                - 0.70 * ep(e, "e_conflict")
            ),
            "HPEM": (
                1.15 * ep(e, "e_directional_pressure")
                + 1.25 * ep(e, "e_cascade")
                + 0.85 * ep(e, "e_pressure_strength")
                + 0.35 * ep(e, "e_volatility")
                + 0.25 * ep(e, "e_jump")
                + 0.35 * ep(e, "e_shock_aligned_plie")
                - 0.65 * ep(e, "e_absorption")
                - 0.75 * ep(e, "e_conflict")
            ),
            "RHA": (
                1.10 * ep(e, "e_directional_pressure")
                + 1.20 * ep(e, "e_rejection")
                + 0.70 * ep(e, "e_absorption")
                + 0.80 * ep(e, "e_rha_strict")
                + 0.35 * ep(e, "e_shock_against_plie") * max(ep(e, "e_rejection"), ep(e, "e_absorption"))
                - 0.50 * ep(e, "e_cascade")
                - 0.70 * ep(e, "e_neutral_context")
                - 0.40 * ep(e, "e_quality_bad")
            ),
            "VT": (
                1.25 * ep(e, "e_active")
                + 0.60 * ep(e, "e_volatility")
                + 0.45 * ep(e, "e_trend_strength")
                + 0.45 * ep(e, "e_neutral_or_mixed")
                + 0.40 * ep(e, "e_shock_active_breakout")
                - 0.45 * ep(e, "e_directional_pressure") * ep(e, "e_pressure_strength")
            ),
            "RC": (
                1.25 * ep(e, "e_quiet")
                + 0.55 * ep(e, "e_neutral_context")
                + 0.45 * ep(e, "e_low_vol")
                + 0.35 * ep(e, "e_range_compression")
                + 0.30 * ep(e, "e_low_trend")
                - 0.65 * ep(e, "e_conflict")
                - 0.45 * ep(e, "e_pressure_strength")
            ),
            "AMB": (
                1.25 * ep(e, "e_conflict")
                + 0.90 * ep(e, "e_quality_bad")
                + 0.45 * ep(e, "e_mixed_context")
                + 0.45 * ep(e, "e_cross_window_conflict")
                + 0.30 * ep(e, "e_label_margin_low")
                - 0.35 * ep(e, "e_quiet")
            ),
        }
        if prev_state == "HPEM":
            logits["RHA"] += 0.80 * ep(e, "e_rejection") + 0.55 * ep(e, "e_absorption") - 0.45 * ep(e, "e_cascade")
            logits["RHA"] += self._w("target", "HPEM_to_RHA_reverse_shock", 0.90) * ep(e, "e_shock_against_plie") * max(ep(e, "e_rejection"), ep(e, "e_absorption"))
            logits["ST"] += 0.30 * ep(e, "e_baseline")
        elif prev_state == "RC":
            logits["VT"] += 0.80 * ep(e, "e_active") + 0.35 * ep(e, "e_volatility")
            logits["VT"] += self._w("target", "RC_to_VT_active_shock", 0.85) * ep(e, "e_shock_active_breakout")
            logits["HPEM"] += 0.40 * ep(e, "e_strong_entry") + 0.40 * ep(e, "e_cascade")
            logits["HPEM"] += self._w("target", "RC_to_HPEM_aligned_shock", 0.80) * ep(e, "e_shock_aligned_plie") * ep(e, "e_directional_pressure")
        elif prev_state == "ST":
            logits["HPEM"] += 0.70 * ep(e, "e_cascade") + 0.35 * ep(e, "e_accel")
            logits["HPEM"] += self._w("target", "ST_to_HPEM_aligned_shock", 0.75) * ep(e, "e_shock_aligned_plie") * ep(e, "e_cascade")
            logits["RHA"] += 0.55 * ep(e, "e_rha_strict") + 0.25 * ep(e, "e_rejection")
            logits["RHA"] += self._w("target", "ST_to_RHA_reverse_shock", 0.70) * ep(e, "e_shock_against_plie") * ep(e, "e_rha_strict")
        elif prev_state == "RHA":
            logits["VT"] += 0.55 * ep(e, "e_active") * (1 - ep(e, "e_directional_pressure"))
            logits["VT"] += self._w("target", "RHA_to_VT_active_shock", 0.75) * ep(e, "e_shock_active_breakout") * ep(e, "e_neutral_or_mixed")
            logits["RC"] += 0.45 * ep(e, "e_quiet") + 0.30 * ep(e, "e_neutral_context")
            logits["HPEM"] += self._w("target", "RHA_to_HPEM_rebreak", 0.65) * ep(e, "e_shock_aligned_plie") * ep(e, "e_cascade")
        elif prev_state == "AMB":
            clarity = 1 - max(ep(e, "e_conflict"), ep(e, "e_quality_bad"))
            for s in ["ST", "VT", "RC", "RHA", "HPEM"]:
                logits[s] += 0.35 * clarity

        for s in self.states:
            if s != "AMB":
                logits[s] -= 0.35 * ep(e, "e_quality_bad")
            if s == prev_state:
                logits[s] = -1e9

        arr = np.array([logits[s] for s in self.states], dtype=float)
        arr = arr - np.nanmax(arr)
        p = np.exp(arr)
        p = np.where(np.isfinite(p), p, 0.0)
        if p.sum() <= 0:
            p = np.ones_like(p)
        return p / p.sum()

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        priors = df[[f"prior_{s}" for s in self.states]].to_numpy(float)
        priors = np.maximum(priors, self.emission_floor)
        priors = priors / priors.sum(axis=1, keepdims=True)
        priors = np.power(priors, self.emission_power)
        priors = priors / priors.sum(axis=1, keepdims=True)

        e_cols = [c for c in df.columns if c.startswith("e_")]
        e_arr = {c: pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float) for c in e_cols}

        def e_dict_at(t: int) -> dict[str, float]:
            return {c: float(arr[t]) for c, arr in e_arr.items()}

        def delta_dict_at(t: int) -> dict[str, float]:
            return {c: float(arr[t] - arr[t - 1]) for c, arr in e_arr.items()}

        alpha = np.zeros((self.k, self.d_max), dtype=float)
        alpha[:, 0] = priors[0]
        alpha /= alpha.sum()

        belief = np.zeros((n, self.k), dtype=float)
        expected_age = np.zeros(n, dtype=float)
        transition_prob = np.zeros(n, dtype=float)
        candidate = [""] * n

        belief[0] = alpha.sum(axis=1)
        expected_age[0] = 1.0
        candidate[0] = self.states[int(np.argmax(belief[0]))]
        transition_prob[0] = 1.0
        prev_candidate = candidate[0]

        for t in range(1, n):
            e = e_dict_at(t)
            delta = delta_dict_at(t)
            new_alpha = np.zeros_like(alpha)
            exit_masses = np.zeros(self.k, dtype=float)
            avg_hazards = np.zeros(self.k, dtype=float)

            for i, state in enumerate(self.states):
                h = self._hazard_vec(state, e, delta)
                mass = alpha[i]
                exit_masses[i] = float(np.dot(mass, h))
                avg_hazards[i] = exit_masses[i]
                stay = mass * (1 - h)
                new_alpha[i, 1:] += stay[:-1]
                new_alpha[i, -1] += stay[-1]

            for i, prev_state in enumerate(self.states):
                if exit_masses[i] <= 0:
                    continue
                dest = self._target_probs(prev_state, e, delta)
                new_alpha[:, 0] += exit_masses[i] * dest
                new_alpha[i, 0] -= exit_masses[i] * dest[i]  # dest[i] should be zero-like, safe

            new_alpha = np.maximum(new_alpha, 0.0)
            new_alpha *= priors[t][:, None]
            s = float(new_alpha.sum())
            if not np.isfinite(s) or s <= 0:
                new_alpha = np.zeros_like(alpha)
                new_alpha[:, 0] = priors[t]
                s = float(new_alpha.sum())
            alpha = new_alpha / s

            belief[t] = alpha.sum(axis=1)
            expected_age[t] = float((alpha * self._ages[None, :]).sum())
            cand_i = int(np.argmax(belief[t]))
            cand = self.states[cand_i]
            candidate[t] = cand

            prev_i = self.states.index(prev_candidate)
            if cand != prev_candidate:
                dest = self._target_probs(prev_candidate, e, delta)
                hbar = np.clip(avg_hazards[prev_i] / max(belief[t - 1, prev_i], 1e-9), 0, 1)
                transition_prob[t] = float(hbar * dest[cand_i])
            else:
                hbar = np.clip(avg_hazards[prev_i] / max(belief[t - 1, prev_i], 1e-9), 0, 1)
                transition_prob[t] = float(1 - hbar)
            prev_candidate = cand

        out = df.copy()
        for i, state in enumerate(self.states):
            out[f"b_{state}"] = belief[:, i]
        out["candidate_state"] = candidate
        out["candidate_confidence"] = belief.max(axis=1)
        sorted_b = np.sort(belief, axis=1)
        out["belief_gap"] = sorted_b[:, -1] - sorted_b[:, -2]
        out["belief_entropy"] = -(belief * np.log(np.maximum(belief, 1e-12))).sum(axis=1)
        out["expected_state_age"] = expected_age
        out["candidate_transition_probability"] = transition_prob
        return out
