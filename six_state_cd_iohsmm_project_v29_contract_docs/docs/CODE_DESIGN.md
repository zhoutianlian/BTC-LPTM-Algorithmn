# 代码设计文档

## 1. 项目结构

```text
six_state_cd_iohsmm_project/
  configs/
    feature_contract.yaml
    model_config.yaml
    state_definitions.yaml
  six_state_engine/
    config.py
    constants.py
    data_loader.py
    validation.py
    canonical.py
    transforms.py
    evidence.py
    filter.py
    stability.py
    evaluation.py
    visualization.py
    pipeline.py
    cli.py
  docs/
    ALGORITHM_DESIGN.md
    EXECUTION_MANUAL.md
    CODE_DESIGN.md
  run_pipeline.py
  requirements.txt
```

## 2. 模块职责

### `data_loader.py`

- 接受输入 `.zip` 或输入目录；
- `.zip` 输入会解压到输出目录下的 `_extracted_input/`，目录输入会直接递归扫描；
- 自动识别 CSV 文件角色；
- 统一字段名为 lower snake case。

### `validation.py`

- 检查 forbidden future columns；
- 检查 required columns；
- 检查 available_time <= time；
- 输出 no-leakage report。

### `canonical.py`

- 构造主时钟；
- exact join PLIE/HMM/memory/path；
- backward-asof join price/vol/FHMV；
- 派生质量和 stale flags；
- 输出 canonical feature frame。

### `evidence.py`

- 拟合 train-only robust scaler；
- 构造 semantic evidence primitives；
- 生成六状态 score 和 soft prior。

### `filter.py`

- 实现 CD-IOHSMM-lite online filtered inference；
- 维护 state-age alpha matrix；
- 输出 raw belief vector。

### `stability.py`

- 实现 EWMA stable belief；
- 执行 hysteresis、minimum duration、state-specific gate；
- 输出 stable state 与 transition reason。

### `evaluation.py`

- 状态特征统计；
- 状态段统计；
- 典型样本；
- transition events；
- 指定场景审查；
- validation report。

### `visualization.py`

- 生成交互式 HTML dashboard；
- 展示价格、belief、evidence、状态分布和切换事件。

## 3. 输入输出边界

输入：

```text
input.zip 或 model_input/
```

输出：

```text
outputs/<run_name>/*.csv
outputs/<run_name>/*.json
outputs/<run_name>/*.md
outputs/<run_name>/six_state_dashboard.html
```

## 4. 可扩展点

1. 增加多尺度 path absorption：
   - 在 canonical 中追加 `path_context_6h/12h/48h`；
   - 在 evidence 中增加多尺度加权；
   - 不改变输出接口。

2. 增加 constrained EM：
   - 用当前 `prior_STATE` 和 anchor 样本初始化 emission；
   - 在 train block 内训练；
   - validation/test/live 仍使用 filtered inference。

3. 增加更复杂 emission：
   - Student-t emission；
   - monotonic GAM；
   - EBM；
   - LightGBM with monotonic constraints。

## 5. 审计原则

- 任何未来收益、当前事件未来 response、offline smoothed posterior 都不得进入 live output。
- 所有 scaler / threshold 应只在 train split 拟合。
- 输出应保留中间 evidence，方便人工审查金融语义。

---

# V2.2 代码设计：Responsiveness Layer

## 修改模块

```text
six_state_engine/evidence.py
    新增 1h shock evidence。

six_state_engine/filter.py
    将 shock evidence 加入 state-specific exit hazard 与 input-dependent target transition。

six_state_engine/stability.py
    新增 State Invalidity / Exit Pressure 与 Short-horizon Price Shock Fast Lane。

six_state_engine/evaluation.py
    新增 responsiveness_diagnostics。

six_state_engine/visualization.py
    dashboard 展示 shock / exit pressure / invalidity / dynamic threshold。

configs/model_config.yaml
    新增 responsiveness 配置组。
```

## 核心实现边界

- 不改变六状态定义；
- 不取消 EWMA、hysteresis、minimum duration；
- 不使用未来价格；
- 不让单根 bar 直接决定状态；
- 所有阈值配置化；
- 输出所有关键中间变量，便于人工审查与回滚。

## Stability switch path

稳定层现在有三条切换路径：

```text
normal_entry:
    新状态 stable belief >= tau_in 且 gap 满足。

exit_pressure:
    旧状态已被连续证据证伪，或旧状态 stable belief <= tau_out。

price_shock_fast_lane:
    高质量 1h 冲击 + 目标状态金融映射成立 + gate 成立。
```

三条路径均要求通过 target gate 与 minimum duration。

## V2.3 代码变更

新增模块：

```text
six_state_engine/segment_quality.py
```

核心函数：

```text
apply_state_confirmation(df, model_config)
compute_segment_diagnostics(df, model_config)
save_segment_quality_outputs(output_dir, df, model_config)
```

`pipeline.py` 在 production stability layer 后调用 `apply_state_confirmation`，然后输出新增确认/交易质量字段。`evaluation.py` 在常规报告生成阶段调用 `save_segment_quality_outputs`，生成状态段诊断与质量审计文件。`visualization.py` 改为轻量化 Plotly + 右侧 inspection panel 结构，避免把所有详细内容塞进 Plotly hover，从而提升全量历史数据的可检查性。


## V2.4：Directional Quality Layer / 方向生命周期层

本版本新增 `directional_quality.py`，在不改变六状态 `stable_state` 的前提下，增加 HPEM/RHA/VT/ST 的交易方向解释层：

- HPEM：检查是否顺 PLIE 压力方向穿透，并输出 aligned move、MFE、giveback、decay flag。
- RHA：区分 `RHA_REVERSAL_UP/DOWN`、`RHA_STALL`、`RHA_WEAK` 与 `INVALID_RHA_FOLLOWS_PLIE`。
- VT：区分 `VT_UP`、`VT_DOWN` 与 `VT_MIXED`。
- ST：输出 `ST_UP/DOWN/WEAK_ST`。

新增字段包括 `directional_state`、`trade_direction`、`direction_confidence`、`directional_trading_quality`、`directional_no_trade_reason`、`directional_exit_trigger_flag`。

本层只使用状态进入以来截至当前时点已经发生的价格路径与当前可得的 PLIE/path/price context，不使用未来状态段终点回填当前结果。详细说明见 `docs/V2_4_DIRECTIONAL_QUALITY.md`。

## V2.5 code additions

- `six_state_engine/price_expectation.py`: causal row-level price-expectation compliance and trading-quality adjustment.
- `six_state_engine/stability.py`: V2.5 integrates price-expectation invalidity into live exit pressure. Strong causal violations can lower effective hysteresis thresholds and use a semantically gated target override.
- `six_state_engine/segment_quality.py`: segment-level price-expectation audit flags.
- `six_state_engine/visualization.py`: dashboard inspection panel and diagnostic traces for price expectation.

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
