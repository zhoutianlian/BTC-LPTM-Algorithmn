# Six-State Engine Validation Report

- Rows: 39865
- Time range: 2021-08-28 11:50:00 → 2026-04-15 06:50:00
- No-leakage passed: True
- Switch rate per row: 0.077361

## State Distribution
- ST: 11.7672%
- VT: 17.6370%
- RC: 20.1555%
- RHA: 25.5613%
- HPEM: 24.3672%
- AMB: 0.5117%

## Duration Summary
- AMB: segments=29, mean_steps=7.03, median_steps=6.00, short_1_step_ratio=0.00%
- HPEM: segments=689, mean_steps=14.10, median_steps=10.00, short_1_step_ratio=3.19%
- RC: segments=484, mean_steps=16.60, median_steps=12.00, short_1_step_ratio=0.00%
- RHA: segments=746, mean_steps=13.66, median_steps=11.00, short_1_step_ratio=0.00%
- ST: segments=589, mean_steps=7.96, median_steps=6.00, short_1_step_ratio=0.00%
- VT: segments=547, mean_steps=12.85, median_steps=10.00, short_1_step_ratio=0.00%

## Responsiveness Diagnostics
- shock_event_count: 2994
- shock_event_ratio: 0.075103
- shock_switch_within_1_step: 0.260521
- shock_switch_within_3_steps: 0.372077
- shock_switch_within_6_steps: 0.480628
- fast_transition_count: 146
- exit_pressure_trigger_count: 1439
- delayed_hold_count: 3941
- candidate_stable_mismatch_ratio: 0.188862
- mean_exit_pressure: 0.279281
- p95_exit_pressure: 0.667929

## Segment Quality Summary
- Segments: 3084
- q_short_low_move: 338
- q_rc_trending: 14
- q_hpem_weak_price_impact: 0
- q_hpem_directional_decay: 584
- q_rha_invalid_direction: 51
- q_vt_weak_movement: 30
- q_vt_weak_trend: 119
- q_vt_mixed_direction: 313
- q_st_weak_trend: 346
- q_low_trading_quality: 2791
- q_low_directional_quality: 2791
- q_low_price_expectation_quality: 2791
- q_price_expectation_exit: 1825
- q_rc_price_too_large: 351
- q_vt_low_move_price: 493
- q_st_price_expectation: 569
- q_hpem_price_expectation_exit: 0
- q_rha_price_expectation_exit: 0
- q_price_expectation_low_score: 1457
- q_hpem_px_failure: 314
- q_rha_px_failure: 711
- q_rc_price_not_small: 102
- q_vt_low_move_px: 462
- q_vt_mixed_or_weak: 526
- q_st_price_expectation_fail: 569
- q_st_flat_or_extreme: 201
### By State
- AMB: segments=29, median_abs_return_bps=46.49, median_range_bps=85.36, mean_trading_quality=0.301, weak_segment_ratio=6.90%
- HPEM: segments=689, median_abs_return_bps=62.18, median_range_bps=124.99, mean_trading_quality=0.430, weak_segment_ratio=1.45%
- RC: segments=484, median_abs_return_bps=85.32, median_range_bps=137.67, mean_trading_quality=0.234, weak_segment_ratio=5.99%
- RHA: segments=746, median_abs_return_bps=59.54, median_range_bps=118.61, mean_trading_quality=0.120, weak_segment_ratio=18.36%
- ST: segments=589, median_abs_return_bps=46.48, median_range_bps=78.77, mean_trading_quality=0.086, weak_segment_ratio=7.47%
- VT: segments=547, median_abs_return_bps=66.86, median_range_bps=131.75, mean_trading_quality=0.112, weak_segment_ratio=31.26%

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
