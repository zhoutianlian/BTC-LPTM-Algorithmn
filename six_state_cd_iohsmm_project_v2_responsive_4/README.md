# Six-State CD-IOHSMM Market State Engine

这是一个面向 BTC 清算—价格反馈链的六状态市场状态工程项目。

项目实现内容：

- 从 `input.zip` 自动解压并识别 CSV 文件。
- 构造严格 no-leakage 的 canonical feature frame。
- 生成金融语义 evidence primitives。
- 使用 SSEM-initialized CD-IOHSMM-lite 做 online filtered inference。
- 输出六状态 belief vector、candidate state、stable state、duration、transition reason。
- 生成状态统计、典型样本、校验报告和交互式 HTML 可视化。
- 提供算法文档、执行手册和代码设计文档。

## 快速运行

```bash
pip install -r requirements.txt

python run_pipeline.py \
  --input-zip /path/to/input.zip \
  --output-dir outputs/run_latest
```

运行后重点查看：

```text
outputs/run_latest/state_timeseries.csv
outputs/run_latest/state_feature_stats.csv
outputs/run_latest/typical_scenarios.csv
outputs/run_latest/transition_events.csv
outputs/run_latest/validation_report.md
outputs/run_latest/no_leakage_report.json
outputs/run_latest/six_state_dashboard.html
```

## 当前版本说明

当前版本是 `SSEM-initialized CD-IOHSMM-lite`：

- emission 由金融语义 evidence layer 与 softmax prior 给出；
- transition 使用 input-dependent transition mask；
- duration 使用 state-specific exit hazard；
- 推理使用 online filtered HSMM recursion；
- 输出使用 EWMA + hysteresis + minimum duration 的 production stability layer。

该版本保留未来升级到 full constrained EM / posterior regularization 的工程接口，但不会在无人工标签下随机训练无约束隐状态，避免 label switching 和语义漂移。

## V2.2 Responsiveness Optimization

This version adds targeted responsiveness controls without changing the six-state definitions:

- `State Invalidity / Exit Pressure`: repeated causal evidence against the current stable state lowers retention and activates `tau_out`.
- `Short-horizon Price Shock Fast Lane`: high-quality 1h shocks can temporarily lower hysteresis only for financially compatible target states.

Disable or rollback via:

```yaml
responsiveness:
  enabled: false
```

Key audit fields in `state_timeseries.csv`:

```text
state_invalidity_score
exit_pressure_current_state
fast_transition_flag
exit_pressure_trigger_flag
delayed_hold_flag
effective_tau_in
effective_belief_gap_min
effective_ewma_lambda
```

## V2.3：状态确认、交易触发质量与 Dashboard Hover 优化

本版本在不改变六状态主定义、不覆盖 `stable_state` 的前提下，新增了稳定状态之后的因果确认层：`Causal State Confirmation + Trading Trigger Quality Layer`。它只使用当前状态段从进入时点到当前时点已经发生的信息，输出 `state_confirmation_status`、`state_confirmation_score`、`weak_state_flag`、`trading_trigger_quality` 和 `trading_trigger_state`，用于区分“状态识别标签”和“是否适合作为交易触发语境”。

新增状态段审计输出：

```text
segment_diagnostics.csv
segment_quality_flags.csv
low_quality_segment_examples.csv
segment_quality_summary.json
```

增强版 Dashboard 将长 hover 内容转移到右侧 inspection panel。Plotly hover 本身只保留短提示，从而提高历史数据 hover 命中率；详细内容包含日期、价格、状态确认、交易质量、PLIE/HMM、path、price context、responsiveness、belief vector 和 transition reason。


## V2.4：Directional Quality Layer / 方向生命周期层

本版本新增 `directional_quality.py`，在不改变六状态 `stable_state` 的前提下，增加 HPEM/RHA/VT/ST 的交易方向解释层：

- HPEM：检查是否顺 PLIE 压力方向穿透，并输出 aligned move、MFE、giveback、decay flag。
- RHA：区分 `RHA_REVERSAL_UP/DOWN`、`RHA_STALL`、`RHA_WEAK` 与 `INVALID_RHA_FOLLOWS_PLIE`。
- VT：区分 `VT_UP`、`VT_DOWN` 与 `VT_MIXED`。
- ST：输出 `ST_UP/DOWN/WEAK_ST`。

新增字段包括 `directional_state`、`trade_direction`、`direction_confidence`、`directional_trading_quality`、`directional_no_trade_reason`、`directional_exit_trigger_flag`。

本层只使用状态进入以来截至当前时点已经发生的价格路径与当前可得的 PLIE/path/price context，不使用未来状态段终点回填当前结果。详细说明见 `docs/V2_4_DIRECTIONAL_QUALITY.md`。

## V2.5 update: Price Expectation Compliance

The engine now includes a V2.5 causal price-expectation compliance layer. It keeps `stable_state` as the market-structure state and adds row-level diagnostics and trading-quality adjustments that answer: *does the price path inside the current state segment still match this state's expected behavior?*

See `docs/V2_5_PRICE_EXPECTATION_COMPLIANCE.md` for details.


### V2.5 live exit pressure

Strong price-expectation failures can now feed the production stability layer as targeted exit pressure. To disable this while keeping diagnostics, set `state_price_expectation.use_in_stability: false`.

## V2.5 Update: Price Expectation Compliance

V2.5 adds a causal state price-behavior compliance layer. It keeps the six-state `stable_state` framework intact, but continuously checks whether the current state segment's observed price path still matches the expected behavior of that state.

New outputs include:

```text
state_price_expectation_score
state_price_expectation_issue
price_expectation_exit_flag
price_expectation_exit_target
price_expectation_exit_pressure
execution_state
execution_quality
```

The dashboard now includes a dedicated price-expectation inspection section, including HPEM giveback, RHA subtype, VT direction, RC drift, VT low-move and ST flat/weak/extreme flags.

See `docs/V2_5_PRICE_EXPECTATION_COMPLIANCE.md` for details.
