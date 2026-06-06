# 执行手册

## 1. 环境准备

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

## 2. 运行命令

```bash
python run_pipeline.py \
  --input-zip ../input.zip \
  --output-dir outputs/run_latest
```

## 3. 输出文件

| 文件 | 说明 |
|---|---|
| `canonical_features.csv` | no-leakage canonical frame |
| `semantic_evidence.csv` | 中间 semantic evidence、state score 与 prior |
| `state_timeseries.csv` | 主输出：belief、candidate、stable_state、duration、transition reason |
| `state_timeseries_full_debug.csv` | 完整调试输出 |
| `state_definitions.csv` | 六状态编号、名称、含义 |
| `state_feature_stats.csv` | 每个 stable state 的核心特征统计 |
| `typical_scenarios.csv` | 每个状态的典型样本 |
| `state_duration_segments.csv` | 连续状态段与持续时间 |
| `transition_events.csv` | 状态切换事件与原因 |
| `no_leakage_report.json` | 字段契约与未来函数检查 |
| `validation_report.json` | 校验报告 JSON |
| `validation_report.md` | 校验报告 Markdown |
| `six_state_dashboard.html` | 交互式 HTML 可视化 |

## 4. 如何检查结果

建议按以下顺序检查：

1. 打开 `no_leakage_report.json`，确认 `passed=true`。
2. 查看 `state_distribution`，确认没有单一状态塌缩。
3. 查看 `state_feature_stats.csv`：
   - HPEM 应有较高 cascade / pressure；
   - RHA 应有较高 rejection / absorption；
   - VT 应有较高 active / volatility；
   - RC 应有较高 quiet / neutral；
   - AMB 应有较高 conflict / quality_bad；
   - ST 应有较高 baseline。
4. 打开 `six_state_dashboard.html`，检查价格路径、belief、evidence 和切换事件。
5. 查看 `validation_report.md` 中两个指定场景：
   - 2026-04-03 到 2026-04-04 应偏 RHA；
   - 2026-01-29 到 2026-01-31 应偏 VT。

## 5. 常见错误

### 关键字段缺失

工程会明确抛出类似：

```text
Missing required columns: ...
```

处理方式：检查 CSV 是否与当前特征契约兼容，或在 `feature_contract.yaml` 中添加字段映射/候选名。

### no-leakage 检查失败

如果禁止列进入 canonical frame，工程会中断。不要绕过该检查，应检查字段选择逻辑。

### HTML 太大

如果数据量很大，dashboard 会自动下采样绘图，不影响 CSV 主输出。

## 6. 参数调整

主要参数位于：

```text
configs/model_config.yaml
configs/feature_contract.yaml
```

不建议为了单次样本内效果随意修改状态定义或 evidence 方向。

---

# V2.2 响应性优化执行与审查

## 运行

执行方式不变：

```bash
python run_pipeline.py --input-zip /path/to/input.zip --output-dir outputs/run_latest
```

## 配置

新增配置位于 `configs/model_config.yaml`：

```yaml
responsiveness:
  enabled: true
  price_shock:
    enabled: true
    min_price_shock_1h: 0.72
    min_short_impulse_1h: 0.45
    min_trend_strength_1h: 0.45
    max_price_missing_ratio_1h: 0.20
    block_on_price_gap: true
    shock_lambda: 0.55
    tau_discount: {...}
    gap_discount: 0.03
    local_cooldown_steps: 1
  exit_pressure:
    enabled: true
    decay: 0.55
    soft_threshold: 0.45
    hard_threshold: 0.65
    exit_lambda: 0.58
    tau_discount: {...}
    gap_discount: 0.03
```

可以一键回滚：

```yaml
responsiveness:
  enabled: false
```

## 审查指标

`validation_report.json` 中新增 `responsiveness_diagnostics`，重点看：

```text
shock_event_count
shock_switch_within_1_step / 3_steps / 6_steps
fast_transition_count
exit_pressure_trigger_count
delayed_hold_count
candidate_stable_mismatch_ratio
mean_exit_pressure / p95_exit_pressure
```

优化目标不是提高总切换频率，而是减少明显滞后。若出现以下情况，需要回调参数：

```text
1-step state segment 比例明显增加；
HPEM/RHA/VT 来回抖动；
AMB 大幅上升并重新吞掉 RC；
所有 shock 都被打成 HPEM；
普通价格噪声触发大量 fast_transition_flag。
```

## V2.3 输出检查

运行完成后，建议按以下顺序检查：

```text
1. state_timeseries.csv
   检查 stable_state、state_confirmation_status、weak_state_flag、trading_trigger_quality。

2. segment_diagnostics.csv
   检查每个连续状态段的入场价、出场价、净收益、区间高低、MFE/MAE、均值 price context 和 dominant path evidence。

3. segment_quality_flags.csv
   快速定位短持续低位移、RC 高趋势、HPEM 弱价格冲击、VT/ST 弱趋势等问题段。

4. low_quality_segment_examples.csv
   人工复核优先级最高的问题段。

5. six_state_dashboard.html
   使用右侧 inspection panel 检查日期、PLIE/HMM、path、price context、belief、transition reason 和 trading quality。
```

Dashboard 支持拖拽缩放、Start/End 时间选择、快捷时间窗口和价格 Y 轴随所选时间段自适应。


## V2.4：Directional Quality Layer / 方向生命周期层

本版本新增 `directional_quality.py`，在不改变六状态 `stable_state` 的前提下，增加 HPEM/RHA/VT/ST 的交易方向解释层：

- HPEM：检查是否顺 PLIE 压力方向穿透，并输出 aligned move、MFE、giveback、decay flag。
- RHA：区分 `RHA_REVERSAL_UP/DOWN`、`RHA_STALL`、`RHA_WEAK` 与 `INVALID_RHA_FOLLOWS_PLIE`。
- VT：区分 `VT_UP`、`VT_DOWN` 与 `VT_MIXED`。
- ST：输出 `ST_UP/DOWN/WEAK_ST`。

新增字段包括 `directional_state`、`trade_direction`、`direction_confidence`、`directional_trading_quality`、`directional_no_trade_reason`、`directional_exit_trigger_flag`。

本层只使用状态进入以来截至当前时点已经发生的价格路径与当前可得的 PLIE/path/price context，不使用未来状态段终点回填当前结果。详细说明见 `docs/V2_4_DIRECTIONAL_QUALITY.md`。

## V2.5 execution notes

Run the pipeline as before. New outputs appear in `state_timeseries.csv`, `segment_diagnostics.csv`, and `six_state_dashboard.html`. To disable the new layer, set:

```yaml
state_price_expectation:
  enabled: false
```

V2.5 can feed strong price-expectation violations into live exit pressure. This can be disabled independently if you want diagnostics/trading-quality only:

```yaml
state_price_expectation:
  use_in_stability: true
```

---

## V2.5 Price Expectation Compliance Layer

The project now includes a causal price-expectation compliance layer. It does not redefine the six market states. It checks whether the observed entry-to-current price path is still consistent with the state’s expected price behavior.

Key new fields:

```text
state_price_expectation_score
state_price_expectation_issue
price_expectation_exit_flag
price_expectation_exit_target
price_expectation_exit_reason
price_expectation_exit_pressure
price_expectation_trading_quality
execution_state
execution_quality
execution_no_trade_reason
```

Core state-specific expectations:

```text
HPEM: price should continue in PLIE direction; no-response, adverse move and giveback weaken/exit HPEM.
RHA : reversal is tradeable; stall is low-quality/no-trade; following PLIE invalidates RHA.
RC  : price movement should be small; drift/range break/trend exclusion weakens/exits RC.
VT  : price movement should be large; mixed direction is allowed as state context but no directional trade.
ST  : mild orderly trend; flat or extreme behavior weakens/exits ST.
```

All checks are causal and use only data available up to the current row.
