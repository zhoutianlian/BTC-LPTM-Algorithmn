# Six-State Engine Validation Report

- Rows: 40508
- Time range: 2021-08-28 11:50:00 → 2026-05-12 02:50:00
- No-leakage passed: True
- Switch rate per row: 0.076997

## State Distribution
- ST: 11.5878%
- VT: 18.3050%
- RC: 20.1935%
- RHA: 25.1605%
- HPEM: 24.0199%
- AMB: 0.7332%

## Duration Summary
- AMB: segments=36, mean_steps=8.25, median_steps=6.50, short_1_step_ratio=2.78%
- HPEM: segments=691, mean_steps=14.08, median_steps=10.00, short_1_step_ratio=3.18%
- RC: segments=494, mean_steps=16.56, median_steps=12.00, short_1_step_ratio=0.00%
- RHA: segments=746, mean_steps=13.66, median_steps=11.00, short_1_step_ratio=0.00%
- ST: segments=590, mean_steps=7.96, median_steps=6.00, short_1_step_ratio=0.00%
- VT: segments=562, mean_steps=13.19, median_steps=10.00, short_1_step_ratio=0.00%

## Responsiveness Diagnostics
- shock_event_count: 3022
- shock_event_ratio: 0.074603
- shock_switch_within_1_step: 0.260093
- shock_switch_within_3_steps: 0.371608
- shock_switch_within_6_steps: 0.479153
- fast_transition_count: 148
- exit_pressure_trigger_count: 1450
- delayed_hold_count: 3993
- candidate_stable_mismatch_ratio: 0.188506
- mean_exit_pressure: 0.280052
- p95_exit_pressure: 0.669798

## Segment Quality Summary
- Segments: 3119
- q_short_low_move: 343
- q_rc_trending: 14
- q_hpem_weak_price_impact: 0
- q_hpem_directional_decay: 585
- q_rha_invalid_direction: 51
- q_vt_weak_movement: 31
- q_vt_weak_trend: 124
- q_vt_mixed_direction: 326
- q_st_weak_trend: 347
- q_low_trading_quality: 2826
- q_low_directional_quality: 2826
- q_low_price_expectation_quality: 2826
- q_price_expectation_exit: 1850
- q_rc_price_too_large: 359
- q_vt_low_move_price: 508
- q_st_price_expectation: 570
- q_hpem_price_expectation_exit: 0
- q_rha_price_expectation_exit: 0
- q_price_expectation_low_score: 1464
- q_hpem_px_failure: 314
- q_rha_px_failure: 711
- q_rc_price_not_small: 102
- q_vt_low_move_px: 477
- q_vt_mixed_or_weak: 541
- q_st_price_expectation_fail: 570
- q_st_flat_or_extreme: 202
### By State
- AMB: segments=36, median_abs_return_bps=36.14, median_range_bps=83.02, mean_trading_quality=0.266, weak_segment_ratio=22.22%
- HPEM: segments=691, median_abs_return_bps=62.18, median_range_bps=124.46, mean_trading_quality=0.430, weak_segment_ratio=1.45%
- RC: segments=494, median_abs_return_bps=84.65, median_range_bps=136.49, mean_trading_quality=0.235, weak_segment_ratio=5.87%
- RHA: segments=746, median_abs_return_bps=59.54, median_range_bps=118.61, mean_trading_quality=0.120, weak_segment_ratio=18.36%
- ST: segments=590, median_abs_return_bps=46.48, median_range_bps=78.74, mean_trading_quality=0.086, weak_segment_ratio=7.46%
- VT: segments=562, median_abs_return_bps=67.33, median_range_bps=133.54, mean_trading_quality=0.113, weak_segment_ratio=30.60%

## Scenario Review
### rha_pressure_rejection
- Note: 向下清算压力持续但价格不跌反升，应偏 RHA / pressure rejection。
- Rows: 48
- Primary state: RHA
- Expected primary: ['RHA']
- Pass expectation: True
- State distribution: {'RHA': 0.7708333333333334, 'ST': 0.125, 'HPEM': 0.10416666666666667}
- Top path context 6h: {'path_directional_weak': 45, 'path_directional_core': 3}
- Top path label 6h: {'path_pressure_rejection': 19, 'path_reversal_takeover': 8, 'path_baseline_transmission': 6, 'path_cascade_transmission': 6, 'path_partial_absorption': 6}
- Top path context 24h: {'path_directional_weak': 48}
- Top path label 24h: {'path_pressure_rejection': 28, 'path_full_absorption_stall': 7, 'path_partial_absorption': 5, 'path_reversal_takeover': 5, 'path_baseline_transmission': 2}

### vt_active_dominance
- Note: 清算压力中性但价格变化巨大，应偏 VT / active dominance，不应偏 RHA。
- Rows: 72
- Primary state: VT
- Expected primary: ['VT']
- Pass expectation: True
- State distribution: {'VT': 0.9305555555555556, 'RC': 0.06944444444444445}
- Top path context 6h: {'path_neutral_pressure': 66, 'path_directional_weak': 6}
- Top path label 6h: {'path_active_dominance_down': 27, 'path_quiet_no_pressure': 20, 'path_normal_active_dominance_down': 11, 'path_normal_active_dominance_up': 8, 'path_reversal_takeover': 2}
- Top path context 24h: {'path_neutral_pressure': 72}
- Top path label 24h: {'path_active_dominance_down': 37, 'path_quiet_no_pressure': 15, 'path_normal_active_dominance_down': 14, 'path_normal_active_dominance_up': 6}
