# 算法设计文档：SSEM-initialized CD-IOHSMM 六状态市场状态引擎

## 1. 目标

本工程实现一个显式概率状态模型，在每个时点输出六状态 belief vector：

\[
b_t=(b_t^{ST}, b_t^{VT}, b_t^{RC}, b_t^{RHA}, b_t^{HPEM}, b_t^{AMB})
\]

输出不是普通分类标签，而是路径依赖、duration-aware、可在线过滤的市场状态置信分布。

## 2. 六状态定义

| 状态 | 含义 |
|---|---|
| ST | 稳定趋势：方向性较明确，传导有序，非极端级联，非高吸收拒绝 |
| VT | 波动趋势：主动交易主导，高波动，PLIE 不是主解释变量 |
| RC | 区间整理：低清算压力、低主动运动、低路径解释力 |
| RHA | 反转或高吸收：清算压力仍在，但价格拒绝该压力 |
| HPEM | 高压力极端运动：清算压力穿透市场并放大 |
| AMB | 模糊或观望：证据冲突、数据质量不足或不适合高置信交易判断 |

## 3. 输入层级

工程将输入 CSV 映射为以下层级：

1. PLIE / 当前清算被动压力层；
2. HMM liquidation regime 层；
3. 已成熟 market response memory 层；
4. 24h path absorption 层；
5. price / volatility context 层；
6. quality / conflict / no-trade 层。

当前 CSV 中 path absorption 的 categorical context / label 实际为 24h，因此工程统一命名为：

```text
path_context_24h
path_label_24h
path_absorption_score_24h
...
```

未来补齐 6h / 12h / 48h 时，可在同一接口下追加。

## 4. No-leakage canonical frame

主时钟采用 `base_context.csv.time`。其他输入按以下规则对齐：

- PLIE / HMM：exact join；
- market response memory：只使用 `absorption_memory.csv` 中已经 matured 的 stale-aware memory；
- path absorption：要求 `available_time <= time`；
- price / vol context：backward-asof join；
- FHMV：optional backward-asof join。

禁止进入模型的列包括：

```text
ret_20m_bps
ret_30m_bps
ret_60m_bps
plie_residual_*
plie_absorption_*
actual_return_bps
aligned_actual_response_bps
response_percentile
absorption_raw
transmission_ratio_raw
```

## 5. Semantic Evidence Model

工程生成以下核心 evidence primitives：

```text
e_directional_pressure
e_pressure_strength
e_baseline
e_cascade
e_absorption
e_rejection
e_active
e_quiet
e_volatility
e_conflict
e_quality_bad
```

这些 evidence 对应金融机制：

- HPEM：directional pressure + cascade + strong pressure；
- RHA：directional pressure + absorption/rejection；
- VT：active dominance + volatility + neutral/mixed pressure；
- RC：quiet + neutral + low volatility；
- AMB：conflict + quality bad + mixed pressure；
- ST：baseline transmission + moderate pressure/volatility + low conflict。

## 6. CD-IOHSMM-lite Filter

当前实现是 SSEM-initialized CD-IOHSMM-lite：

\[
P(z_t\mid x_{1:t})
\]

通过如下递推实现：

1. emission potential 由 semantic prior 给出；
2. state-specific exit hazard 建模 duration；
3. input-dependent transition destination 建模状态切换证据；
4. alpha state-age matrix 输出 filtered belief。

普通 HMM 只建模隐状态转移；本实现显式维护状态年龄：

\[
\alpha_t(k,a)=P(z_t=k,a_t=a,x_{1:t})
\]

状态切换不能只由当前一行决定，而必须通过：

```text
current evidence
previous state
state age
transition-specific evidence gate
```

## 7. Production Stability Layer

raw belief 再通过生产稳定层：

\[
\tilde b_t = \lambda \tilde b_{t-1} + (1-\lambda)b_t
\]

并使用：

```text
hysteresis
minimum duration
state-specific transition gate
AMB/RHA 特殊进入条件
```

得到最终 `stable_state`。

## 8. 当前最小合理假设

1. 当前 path categorical evidence 只有 24h，暂不伪造 6h/12h/48h path label。
2. 当前版本不做随机无监督 EM，以避免 label switching；先使用语义 evidence 作为 emission prior。
3. price / volatility 层使用当前 CSV 已有 `vol_adaptive / trend_pressure / kalman_slope` 以及 past-only 派生收益/波动。
4. market response 只使用 `absorption_memory.csv` 中已成熟 memory，不直接使用 event 当前未来 response。

---

# V2.2 Responsiveness Optimization：状态退出与短期价格冲击快速通道

本版本在不改变六状态定义、不改变主 evidence 方向的前提下，实现两个局部机制：

1. **State Invalidity / Exit Pressure**：当当前 stable state 被连续的因果证据反复证伪时，逐步降低旧状态保留权，并让 `tau_out` 真正参与退出判断。
2. **Short-horizon Price Shock Fast Lane**：当 1h 价格冲击强、数据质量正常、且目标状态有明确金融解释时，临时降低 hysteresis 阈值和 EWMA 惯性。

该层只影响 transition / stability，不会把单根价格波动直接变成 hard label。

## 新增 price shock evidence

从 price context 中派生以下字段，均为 past-only：

```text
e_price_shock_1h
    max(realized_vol_1h, max_jump_z_1h, jump_proxy_1h) 的归一化冲击强度。

e_short_impulse_1h
    e_price_shock_1h * trend_strength_1h * e_shock_data_valid。

e_shock_aligned_plie
    1h 冲击方向与 PLIE direction 同向，支持 HPEM / liquidation-driven cascade。

e_shock_against_plie
    1h 冲击方向与 PLIE direction 反向，支持 RHA / pressure rejection。

e_shock_active_breakout
    neutral/mixed pressure 下的 1h 主动突破，支持 VT。

e_shock_data_valid
    price gap / missing ratio 检查通过；若失败，不能触发 HPEM/VT/RHA 快速通道。
```

## State invalidity / exit pressure

对当前 stable state 计算失效分数：

```text
HPEM invalidity:
    reverse shock against PLIE + absorption/rejection rising + cascade decay

RHA invalidity:
    directional pressure decays + active/quiet takeover + strict RHA fades

RC invalidity:
    short impulse + active dominance + pressure strength + compression loss

VT invalidity:
    trend/vol decay + quiet compression + liquidation cascade emerges

ST invalidity:
    cascade + strict RHA + active high-vol breakout + quiet/conflict

AMB invalidity:
    conflict / data-quality / cross-window uncertainty decays
```

失效分数通过 EWMA 累积：

\[
ExitPressure_t = \rho ExitPressure_{t-1} + (1-\rho) Invalidity_t
\]

当 exit pressure 到达 soft threshold 时，降低 `tau_in / belief_gap / ewma_lambda`；当到达 hard threshold 或旧状态 stable belief 低于 `tau_out` 时，允许 exit-driven switch。

## Fast lane 触发原则

快速通道只在以下情况生效：

```text
数据质量通过；
e_price_shock_1h 和 e_short_impulse_1h 超阈值；
目标状态符合金融映射；
目标状态 gate 仍然通过；
minimum duration 仍然满足；
局部 cooldown 未触发。
```

状态映射：

```text
aligned PLIE shock     -> HPEM
against PLIE shock     -> RHA
neutral/mixed active   -> VT
conflict/data issue    -> AMB
```

RC/ST 不作为 price-shock fast-lane 的低波目标，只能通过正常 entry 或 exit-driven entry 进入。

## 新增输出字段

```text
state_invalidity_score
state_invalidity_reason
exit_pressure_current_state
fast_transition_flag
fast_transition_reason
exit_pressure_trigger_flag
delayed_hold_flag
effective_tau_in
effective_belief_gap_min
effective_ewma_lambda
```

这些字段用于 dashboard 人工审查：为什么旧状态没有切、是否因为 gate 不满足、是否是 exit pressure 触发、是否是 price shock fast lane 触发。

## V2.3 Causal State Confirmation + Trading Trigger Quality Layer

V2.3 在 `stable_state` 之后新增一个因果确认层，但不重新定义六状态，也不覆盖原有稳定状态。其目标是解决短持续 RC/HPEM/VT/ST 状态段的交易解释性问题：状态可以先被识别为某类市场语境，但是否可作为交易触发，需要观察该状态段自进入以来截至当前时点已经出现的价格位移、区间波动、趋势强度、趋势一致性、跳跃、PLIE/path 证据和冲突质量。

该层输出：

```text
state_confirmation_score
state_confirmation_status
state_confirmed_flag
state_pending_confirmation_flag
weak_state_flag
trading_trigger_quality
trading_trigger_state
state_quality_issue
```

关键原则：

```text
1. 不使用未来数据；
2. 不回填修改历史 stable_state；
3. 不改变六状态金融定义；
4. 将状态识别与交易触发质量分离；
5. 对 RC/HPEM/VT/ST 进行更严格的段内因果确认。
```

其中 RC 需要确认低趋势、低波动、区间压缩和低冲突；HPEM 需要确认清算压力与价格冲击/传导；VT 需要确认主动趋势、位移或 range expansion；ST 需要确认有序趋势、趋势一致性和非极端波动。


## V2.4：Directional Quality Layer / 方向生命周期层

本版本新增 `directional_quality.py`，在不改变六状态 `stable_state` 的前提下，增加 HPEM/RHA/VT/ST 的交易方向解释层：

- HPEM：检查是否顺 PLIE 压力方向穿透，并输出 aligned move、MFE、giveback、decay flag。
- RHA：区分 `RHA_REVERSAL_UP/DOWN`、`RHA_STALL`、`RHA_WEAK` 与 `INVALID_RHA_FOLLOWS_PLIE`。
- VT：区分 `VT_UP`、`VT_DOWN` 与 `VT_MIXED`。
- ST：输出 `ST_UP/DOWN/WEAK_ST`。

新增字段包括 `directional_state`、`trade_direction`、`direction_confidence`、`directional_trading_quality`、`directional_no_trade_reason`、`directional_exit_trigger_flag`。

本层只使用状态进入以来截至当前时点已经发生的价格路径与当前可得的 PLIE/path/price context，不使用未来状态段终点回填当前结果。详细说明见 `docs/V2_4_DIRECTIONAL_QUALITY.md`。

## V2.5 Price Expectation Compliance

V2.5 adds a post-stability causal compliance layer. It checks HPEM/RHA directional continuity, RC small-price behavior, VT large volatile movement, and ST mild orderly trend using only state-entry-to-current data. The layer outputs compliance scores, issue flags, suggested exit targets, and trading-trigger quality adjustments without redefining the six states or rewriting `stable_state`.

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
