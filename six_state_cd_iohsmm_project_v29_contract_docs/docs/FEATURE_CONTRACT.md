# Feature Contract：六状态清算—吸收模型的依赖特征说明

> 版本：V2.9 Reader-friendly Feature Contract with Core Column Reference  
> 契约模式：`contract_mode: strict_fixed_input_bundle`  
> 适用范围：六状态清算—吸收市场状态引擎的输入文件、核心字段、因果约束和 no-leakage 审计。

---

## 1. 这份文档解决什么问题

**Feature Contract（特征契约）** 是本项目对输入数据的正式约定。它不只是告诉工程系统“读取哪些 CSV 文件”，更重要的是规定：模型在每一个时点 `t` 允许知道哪些信息、这些信息如何对齐、哪些字段可以进入 live inference、哪些字段必须被拒绝。

在普通机器学习项目里，输入特征有时只是一个因子列表；但在本项目中，输入契约本身就是模型定义的一部分。原因是六状态模型识别的不是简单价格形态，而是 BTC 杠杆市场中的主导机制：清算压力从哪里来，价格是否服从这股压力，市场是在传导、吸收、放大、主动突破，还是证据冲突。若输入特征缺失、错位或泄漏，模型输出的 `stable_state`、`belief vector`、`directional_state` 和 `price_expectation_exit_flag` 的金融语义都会改变。

这份文档连接了三类工作：

1. **数据工程**：保证输入文件、字段、时间戳和可用时间满足约束；
2. **模型推理**：保证 canonical frame 中的每个字段都能映射到可解释金融证据；
3. **策略接入**：保证不同运行之间的输出语义稳定，便于回测、实盘、审计和版本比较。

一句话说：Feature Contract 规定的不是“数据格式”那么简单，而是六状态引擎的**因果信息边界**。

---

## 2. 从可选 fallback 到 strict fixed input bundle

早期版本曾允许部分文件可选，这是因为系统处在迁移阶段：

- 从旧版 `24h-only path layer` 迁移到 `6h / 12h / 24h / 48h multiscale path`；
- 从 `liqprice_features.csv` 中的 legacy price proxy 迁移到完整的 `price_context_features.csv`；
- `features_liq_dataflow.csv`、`liqprice_features.csv`、`price_context_features.csv`、`path_absorption_multiscale.csv` 曾经可能缺失，engine 会用 fallback features 继续运行。

这个做法适合研究探索期，但不适合当前策略接入阶段。原因是 fallback 会让两次运行看到不同的 feature universe：同样的模型配置，在一次运行中可能使用完整 price context，另一次运行中可能退化到 legacy proxy。下游策略如果不知道这件事，就会误以为同一个 `stable_state = HPEM` 在两个版本中含义相同，实际其证据来源已经变了。

因此当前版本采用：

```yaml
contract_mode: strict_fixed_input_bundle
```

它的含义是：

- 所有 required files 必须存在；
- 所有 required columns 必须存在；
- 文件缺失不允许自动填 0；
- 列缺失不允许静默忽略；
- schema 变化必须显式报错；
- 不允许因为某个文件缺失而自动切换到另一套隐含逻辑。

---

## 3. 为什么严格输入契约对策略型量化模型更安全

### 3.1 Reproducibility：结果可复现

同一配置、同一时间段、同一数据版本应得到相同状态语义。若某些输入文件可选，模型可能在不同运行中使用不同证据层，导致 `state distribution`、`transition matrix` 和 `segment_quality` 不再可比。

### 3.2 Auditability：问题可审计

缺失输入应该成为显式错误，而不是让模型暗中降级。fail-fast 能告诉研究员：这次运行的输入不完整，不能与正常版本混在一起比较。

### 3.3 Comparability：版本可比较

walk-forward run、参数版本、模型版本之间的比较，前提是 feature universe 固定。否则状态分布变化可能来自算法改进，也可能只是输入层缺失。

### 3.4 Production safety：实盘安全

实盘系统不能因为某个辅助文件缺失就切换到另一套逻辑。输入契约稳定是 strategy-facing model 的安全约束。

### 3.5 Strategy integration：策略语义稳定

下游策略需要稳定理解 `stable_state`、`belief_gap`、`directional_state`、`trading_trigger_quality` 和 `price_expectation_exit_flag` 的含义。如果输入契约不稳定，策略规则就会建立在不稳定语义上。

---

## 4. 输入特征如何进入六状态模型

输入文件不会直接等于状态。它们先被清洗、对齐、过滤成 **canonical feature frame（规范化特征框架）**。canonical frame 的每一行对应一个 source-clock 时点，且只包含该时点已经可得的信息。

整体流程是：

```text
Required CSV files
→ Canonical Feature Frame
→ Semantic Evidence Layer
→ Six-state belief inference
→ Production Stability
→ Directional Quality
→ Price Expectation Compliance
→ Strategy-facing outputs
```

也就是说，原始字段会先被压缩成金融语义证据，例如 directional pressure、cascade、absorption、rejection、active dominance、quiet、conflict 和 quality bad。然后这些 evidence 影响六状态 belief vector，belief 再经过生产稳定层生成 `stable_state`。策略侧不应直接交易某个单一输入字段，也不应只凭状态名称交易。

---

## 5. Required Files 总览

| Role | File pattern | Required | 主要提供的信息 | 回答的市场问题 | 对六状态模型的作用 | 缺失风险 |
|---|---|---:|---|---|---|---|
| `base_context` | `base_context.csv` | true | source-clock、price、split、基础 HMM/PLIE context | 模型在哪个 source-clock 时点推理？当前基本价格、HMM/PLIE 背景是什么？ | 建立 canonical frame 时间骨架，所有其他特征对齐到它 | 主时钟错乱，所有特征对齐与状态段审计失效 |
| `plie` | `plie_predictions_source.csv` | true | PLIE direction / intensity / reliability / transition | 当前 forced liquidation flow 的方向、强度、阶段与可靠性是什么？ | 提供压力源，支持 HPEM/RHA/ST/AMB 的语义判断 | 清算压力源缺失，六状态失去核心解释轴 |
| `absorption_memory` | `absorption_memory.csv` | true | 已成熟 event response memory、absorption、takeover、amplification、conflict | 最近清算事件成熟后，市场是传导、吸收、放大还是反向接管？ | 区分传导、吸收、放大、反向接管 | 无法判断压力是否被市场裁决，HPEM/RHA/AMB 混淆 |
| `path_absorption_multiscale` | `path_absorption_multiscale.csv` | true | 6h/12h/24h/48h path_context、path_label 与 scores | 6h/12h/24h/48h 路径上的压力背景和价格响应结构是什么？ | 多尺度路径语境，决定 HPEM/RHA/VT/RC/ST/AMB 边界 | 短长周期冲突无法识别，状态退出和段质量下降 |
| `path_absorption` | `path_absorption.csv` | true | legacy 24h path compatibility/audit fields | 旧版 24h path 层与新版 multiscale path 是否可追溯、一致或存在冲突？ | 提供 24h 历史连续性和 cross-check | 无法与旧版本结果和旧 path 诊断对齐 |
| `price_context_features` | `price_context_features.csv` | true | realized vol、range compression、trend、jump、price quality | 当前价格行为是否符合状态定义：低波、趋势、冲击、压缩或失效？ | 验证价格行为是否符合状态定义，驱动 directional quality 与 price expectation | RC/VT/ST/HPEM/RHA 的价格行为边界失真 |
| `liqprice_features` | `liqprice_features.csv` | true | legacy price/liquidation micro-context auxiliary fields | 旧版 price/liquidation micro-context 是否支持或审计当前压力背景？ | 辅助审计微观清算价格关系，不能主导六状态 | feature universe 不一致，旧版兼容审计缺失 |
| `features_liq_dataflow` | `features_liq_dataflow.csv` | true | liquidation-stress / volatility-stress auxiliary diagnostics | 清算压力和波动压力的辅助 stress 背景是什么？ | 辅助压力背景与异常诊断，不能覆盖主语义层 | stress 辅助诊断缺失，版本可比性下降 |

---

## 6. `base_context.csv`：主时钟与基础上下文

### 文件角色

`base_context.csv` 是 primary source-clock frame，也就是整个 canonical frame 的时间骨架。其他输入文件通常都要对齐到它的 `time`。

### 它回答的市场问题

模型在当前哪个 source-clock 时点推理？这个时点的基础价格、split、HMM/PLIE 背景是什么？

### 核心金融含义

在清算驱动模型里，时间轴不是中性细节。PLIE/HMM 约小时级更新，价格数据可能更高频，market response memory 还需要等 horizon 成熟。如果主时钟错乱，模型就可能在错误时点使用错误证据。

### 对六状态模型的作用

`base_context.csv` 决定每个时点 `t` 模型允许看到的信息集合。它不是普通辅助文件，而是 canonical frame 的基准。

### 与其他特征层的关系

PLIE、absorption memory、path absorption、price context、liqprice 和 FHMV 都需要对齐到 base context 的时间轴。

### 缺失或错误的风险

- source-clock 错乱会导致全局 feature alignment 错误；
- duplicate timestamps 会破坏状态持续时间、状态段审计和 transition 统计；
- price 错误会污染 segment diagnostics 与 price expectation compliance。

### 因果与工程约束

- `time` 必须唯一或可明确去重；
- 不允许无法解释的 duplicate source-clock timestamps；
- 其他特征只能 backward-asof 或 exact join 到该时钟；
- `price` 必须是当前或过去可得价格。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `price` | 当前推理时点的代表性价格。 | 用于状态可视化、状态段收益、price expectation compliance 与 direction layer。 | 由 base context 或价格源提供；只允许使用当前及过去价格。 |
| `split` | 时间切分标识，如 train / validation / test。 | 保证训练、验证、测试按时间顺序隔离，防止随机打乱造成泄漏。 | 由数据准备流程按时间区间生成。 |
| `hmm_state` | HMM liquidation regime 的离散状态编号。 | 提供当前清算压力 regime 背景：强空头清算、轻空头清算、中性、轻多头清算、强多头清算等。 | 由上游 HMM filtered state 产生；在本模型中作为压力语境，不直接等同六状态。 |
| `hmm_conf` | HMM regime 判断置信度。 | 置信度高时清算 regime 更清晰；置信度低时可能提高 AMB 或降低交易触发质量。 | 通常由 HMM posterior 最大值或集中度计算。 |
| `liq_entropy` | 清算 regime posterior 的熵或不确定性。 | 熵高表示多种清算 regime 竞争，证据不清晰，可能提高 AMB / conflict。 | 由 p_state_* 分布计算，例如 -sum(p log p)。 |
| `age_in_state_source` | 上游 HMM/source regime 已持续的时间或 bar 数。 | 帮助判断当前清算压力是新出现还是已经成熟，影响 transition / duration 解释。 | 由上游 source-clock regime 序列累计。 |
| `state_severity` | HMM regime 的压力严重程度。 | 严重程度高说明清算背景更强，可能支持 HPEM/ST/RHA，但需价格响应确认。 | 由 HMM state 或 posterior 与强度映射生成。 |
| `plie_direction` | PLIE 当前清算压力方向，通常 +1 表示空头清算 forced buying 向上压力，-1 表示多头清算 forced selling 向下压力，0 表示方向弱或中性。 | 方向是 HPEM/RHA/ST 判断的核心，但方向压力不是价格结论。 | 由上游 PLIE 模块基于 liquidation flow 与基线响应估计。 |
| `plie_main_bps` | 主 PLIE 位移基线，单位 bps。 | 表示当前 forced liquidation flow 可能施加的被动价格压力幅度；是压力源，不是收益预测。 | 由上游 PLIE 模块在主 horizon 上估计。 |
| `plie_reliability` | PLIE 信号可靠性。 | 可靠性高时 PLIE 更适合作为压力源；可靠性低时价格运动更可能由 active flow 或其他因素主导。 | 由上游 PLIE 模型根据样本质量、清算结构、置信度等计算。 |
| `plie_intensity` | PLIE 清算压力强度。 | 强度高表示当前 forced flow 更显著；需结合价格响应区分 HPEM/RHA。 | 由清算规模、方向性、基线响应等上游指标归一化得到。 |
| `plie_phase` | PLIE 当前阶段，如 stable、accelerating、decaying 等。 | 阶段信息帮助判断压力是否正在扩散、维持或衰减。 | 由上游 PLIE 动态特征生成。 |

---

## 7. `plie_predictions_source.csv`：当前清算压力证据

### 文件角色

`plie_predictions_source.csv` 是当前 PLIE（Passive Liquidation Implied Effect）证据的核心来源。PLIE 描述当前 forced liquidation flow 对价格形成的被动方向压力。

### 它回答的市场问题

当前清算压力来自哪一侧？强度有多大？可靠性如何？这股 forced flow 是刚进入、正在加速、还是正在衰减？

### 核心金融含义

清算不是普通交易。普通交易反映交易者主动意愿，清算则是保证金约束触发后的 forced flow。`plie_direction = +1` 表示空头清算 forced buying，向上压力；`plie_direction = -1` 表示多头清算 forced selling，向下压力。

但 PLIE 是压力源，不是价格预测。高 PLIE 不等于 HPEM。只有当高 PLIE 与价格顺压力方向穿透、path cascade、market response amplification 同时出现时，才更支持 HPEM。若高 PLIE 伴随价格拒绝或反向接管，则更支持 RHA；若证据冲突，则可能提高 AMB。

### 对六状态模型的作用

- 支持 HPEM：强 directional PLIE + cascade + price impact；
- 支持 RHA：directional PLIE 仍在，但价格拒绝；
- 支持 ST：中等 PLIE + baseline/partial transmission；
- 支持 AMB：PLIE 强但 response/path/price 互相冲突。

### 与其他特征层的关系

PLIE 必须与 market response memory、multiscale path absorption 和 price context 一起解释。单看 PLIE 只知道“压力来了”，不知道“市场是否服从压力”。

### 缺失或错误的风险

PLIE 缺失会让模型失去清算压力主轴；方向约定错误会把 HPEM/RHA/ST 的金融含义完全反转。

### 因果与工程约束

PLIE passive horizon 字段只能是当前可得的基线投影，不能包含未来实际收益、residual 或 absorption diagnostic。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `p_state_1` | HMM liquidation regime 中 state 1 的 filtered posterior 概率。 | 描述当前清算 regime 的不确定性，而不是只依赖 hard state；帮助识别压力方向、强度与 AMB 质量风险。 | 由上游 HMM liquidation regime 模块在当前时点 filtered inference 得到；不能使用 smoothed posterior。 |
| `p_state_2` | HMM liquidation regime 中 state 2 的 filtered posterior 概率。 | 描述当前清算 regime 的不确定性，而不是只依赖 hard state；帮助识别压力方向、强度与 AMB 质量风险。 | 由上游 HMM liquidation regime 模块在当前时点 filtered inference 得到；不能使用 smoothed posterior。 |
| `p_state_3` | HMM liquidation regime 中 state 3 的 filtered posterior 概率。 | 描述当前清算 regime 的不确定性，而不是只依赖 hard state；帮助识别压力方向、强度与 AMB 质量风险。 | 由上游 HMM liquidation regime 模块在当前时点 filtered inference 得到；不能使用 smoothed posterior。 |
| `p_state_4` | HMM liquidation regime 中 state 4 的 filtered posterior 概率。 | 描述当前清算 regime 的不确定性，而不是只依赖 hard state；帮助识别压力方向、强度与 AMB 质量风险。 | 由上游 HMM liquidation regime 模块在当前时点 filtered inference 得到；不能使用 smoothed posterior。 |
| `p_state_5` | HMM liquidation regime 中 state 5 的 filtered posterior 概率。 | 描述当前清算 regime 的不确定性，而不是只依赖 hard state；帮助识别压力方向、强度与 AMB 质量风险。 | 由上游 HMM liquidation regime 模块在当前时点 filtered inference 得到；不能使用 smoothed posterior。 |
| `hmm_state` | HMM liquidation regime 的离散状态编号。 | 提供当前清算压力 regime 背景：强空头清算、轻空头清算、中性、轻多头清算、强多头清算等。 | 由上游 HMM filtered state 产生；在本模型中作为压力语境，不直接等同六状态。 |
| `hmm_conf` | HMM regime 判断置信度。 | 置信度高时清算 regime 更清晰；置信度低时可能提高 AMB 或降低交易触发质量。 | 通常由 HMM posterior 最大值或集中度计算。 |
| `liq_entropy` | 清算 regime posterior 的熵或不确定性。 | 熵高表示多种清算 regime 竞争，证据不清晰，可能提高 AMB / conflict。 | 由 p_state_* 分布计算，例如 -sum(p log p)。 |
| `age_in_state_source` | 上游 HMM/source regime 已持续的时间或 bar 数。 | 帮助判断当前清算压力是新出现还是已经成熟，影响 transition / duration 解释。 | 由上游 source-clock regime 序列累计。 |
| `state_pressure_direction` | HMM regime 映射出的清算压力方向。 | 用于确认 forced buying / forced selling 的方向背景。 | 由 hmm_state 或 posterior 加权方向映射产生。 |
| `state_severity` | HMM regime 的压力严重程度。 | 严重程度高说明清算背景更强，可能支持 HPEM/ST/RHA，但需价格响应确认。 | 由 HMM state 或 posterior 与强度映射生成。 |
| `plie_direction` | PLIE 当前清算压力方向，通常 +1 表示空头清算 forced buying 向上压力，-1 表示多头清算 forced selling 向下压力，0 表示方向弱或中性。 | 方向是 HPEM/RHA/ST 判断的核心，但方向压力不是价格结论。 | 由上游 PLIE 模块基于 liquidation flow 与基线响应估计。 |
| `plie_reliability` | PLIE 信号可靠性。 | 可靠性高时 PLIE 更适合作为压力源；可靠性低时价格运动更可能由 active flow 或其他因素主导。 | 由上游 PLIE 模型根据样本质量、清算结构、置信度等计算。 |
| `plie_intensity` | PLIE 清算压力强度。 | 强度高表示当前 forced flow 更显著；需结合价格响应区分 HPEM/RHA。 | 由清算规模、方向性、基线响应等上游指标归一化得到。 |
| `plie_accel_pos` | 当前方向上的清算压力是否加速。 | 加速可支持 HPEM entry 或 ST->HPEM，但若价格拒绝则可能支持 RHA。 | 由上游 PLIE 强度的过去窗口变化计算。 |
| `plie_strong_entry` | 是否刚进入强 PLIE 状态。 | 用于快速识别清算压力突然增强的触发点。 | 由 PLIE intensity / phase / transition 规则生成。 |
| `plie_transition_type` | PLIE 状态转移类型。 | 帮助解释压力是新进入、增强、衰减还是切换。 | 由上游 PLIE 状态机或规则模块输出。 |
| `plie_transition_severity` | PLIE 转移严重程度。 | 严重转移可提高 transition evidence 或 responsiveness。 | 由转移幅度、强度变化、置信度等上游指标生成。 |
| `plie_passive_20m_bps` | 20m horizon 的被动清算压力基线。 | 短 horizon 压力用于短期响应和 fast lane 审查。 | 由 PLIE 模块估计；必须是当前可得的基线投影，不可包含未来实际收益。 |
| `plie_passive_30m_bps` | 30m horizon 的被动清算压力基线。 | 主短期 horizon，用于衡量 forced flow 的预期被动冲击。 | 由 PLIE 模块估计；不是未来收益标签。 |
| `plie_passive_60m_bps` | 60m horizon 的被动清算压力基线。 | 提供稍慢一层的压力基线，帮助识别压力是否有持续性。 | 由 PLIE 模块估计；不可混入未来实际响应。 |
| `plie_main_bps` | 主 PLIE 位移基线，单位 bps。 | 表示当前 forced liquidation flow 可能施加的被动价格压力幅度；是压力源，不是收益预测。 | 由上游 PLIE 模块在主 horizon 上估计。 |

---

## 8. `absorption_memory.csv`：成熟后的市场响应记忆

### 文件角色

`absorption_memory.csv` 是 matured event response memory。它记录最近已经成熟的清算事件之后，市场短期是传导、吸收、放大，还是反向接管。

### 它回答的市场问题

清算事件发生后，市场实际如何裁决这股压力？是继续顺压力走，还是接住、钝化、甚至反向接管？

### 核心金融含义

清算发生在 `t`，并不意味着 `t` 时刻已经知道 `t+30m` 的响应。只有响应窗口成熟之后，这段响应才能进入后续时点的 memory。因此 absorption memory 的核心不是“事后看清楚”，而是“只把已经成熟的信息带入当前”。

### 对六状态模型的作用

- amplification persistence 支持 HPEM；
- absorption persistence 和 takeover count 支持 RHA；
- neutral active strength 支持 VT；
- response conflict 支持 AMB 或降低交易质量。

### 与其他特征层的关系

market response memory 是短期成熟裁决；path absorption 是更长窗口路径语境；PLIE 是当前压力源。三者共同判断压力是否正在穿透、被吸收或失效。

### 缺失或错误的风险

若缺失，模型只能看到当前压力和路径背景，却缺少近期成熟裁决，HPEM/RHA/AMB 容易混淆。若使用未成熟 response，则会造成未来函数泄漏。

### 因果与工程约束

- 只允许 `available_time <= time` 的成熟响应进入；
- current event 的 future response 不能进入 current state；
- stale response 必须降权或标记。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `mar_abs_score_q_staleaware_ewm_6_30m` | 成熟 market response 的吸收分数 EWM。 | 最近已成熟事件中，清算压力是否被市场吸收。高值支持 RHA 或降低 HPEM。 | 基于 available_time 已成熟的 30m response，做质量加权与 stale-aware EWM。 |
| `mar_active_force_aligned_staleaware_ewm_6_30m` | 主动价格力量与清算方向对齐的短期记忆。 | 对齐增强可支持 pressure transmission / HPEM；若反向则支持 RHA。 | 由成熟 response 中主动力量方向与 PLIE 方向关系聚合。 |
| `mar_active_force_price_staleaware_ewm_6_30m` | 成熟响应中的主动价格力量记忆。 | 帮助判断价格运动是否由主动交易接管，而非 PLIE 主导。 | 由成熟事件后的价格反应特征 stale-aware EWM 得到。 |
| `mar_directional_core_freshness_30m` | directional core response memory 的新鲜度。 | 新鲜度高说明短期 response 更可参考；过旧则应降权。 | 由最近成熟 directional response 的 available_time 距当前时间计算。 |
| `mar_directional_core_age_hours_30m` | directional core response memory 的年龄，单位小时。 | 年龄越大，短期响应证据越 stale，可能降低其权重或提高 AMB。 | 由当前 time 与最近成熟 response 时间差计算。 |
| `mar_directional_quality_ewm_6_30m` | 方向性响应质量的短期 EWM。 | 衡量最近成熟 response 是否方向清晰、质量足够。 | 由成熟事件 quality score 聚合。 |
| `mar_takeover_count_12_30m` | 最近窗口内 reversal/takeover 类成熟响应计数。 | 计数上升支持 RHA 或主导权切换。 | 统计已成熟 30m response 中 takeover 标签的数量。 |
| `mar_amplification_persistence_6_30m` | 短期放大持续性。 | 持续放大支持 HPEM；若衰减则可能退出 HPEM。 | 对成熟 response 中 passive amplification 证据聚合。 |
| `mar_absorption_persistence_6_30m` | 短期吸收持续性。 | 持续吸收支持 RHA 或 HPEM->RHA。 | 对成熟 response 中 absorption/stall/rejection 证据聚合。 |
| `mar_neutral_active_strength_evidence_ewm_6_30m` | 中性压力环境下主动运动强度记忆。 | 支持 VT，尤其 PLIE 中性或低解释力时。 | 对成熟 neutral response 中 active move 强度做 EWM。 |
| `mar_neutral_context_persistence_12_30m` | 中性压力环境的持续性记忆。 | 支持 RC/VT 的 pressure source 背景，降低 liquidation-driven 解释。 | 统计或 EWM 聚合近期成熟 neutral context。 |
| `mar_response_conflict_score` | market response 证据冲突分数。 | 短期 response 与 path/PLIE 冲突时提高 AMB 或降低交易质量。 | 由成熟 response 标签、方向、质量与当前路径证据差异计算。 |

---

## 9. `path_absorption_multiscale.csv`：多尺度路径吸收语境

### 文件角色

`path_absorption_multiscale.csv` 是多尺度 path absorption 的核心层。它在 6h / 12h / 24h / 48h 上描述清算压力背景和价格路径响应。

### 它回答的市场问题

过去多个时间尺度上，压力结构是什么？价格是顺压力传导、级联放大、部分吸收、完全停滞、压力拒绝、反向接管，还是 active dominance？

### 核心金融含义

单一窗口不够。6h 反映最近变化，12h 负责中短期确认，24h 是核心 episode context，48h 是慢变量背景。只看 24h 可能让旧压力污染当前 RC/VT；只看 6h 又可能过度噪声。多尺度 path 让模型能识别短长周期之间的迁移与冲突。

### 重要概念解释

- **directional pressure**：窗口内清算压力方向明确；
- **mixed pressure**：多空压力混杂，方向不清；
- **neutral pressure**：清算压力弱或中性；
- **cascade transmission**：压力顺方向穿透并放大，支持 HPEM；
- **baseline transmission**：压力有序传导，支持 ST；
- **partial absorption**：价格仍顺压力，但传导减弱，偏 ST/RHA 过渡；
- **full absorption stall**：压力存在但价格停滞，支持 RHA_STALL；
- **pressure rejection**：价格逆压力方向，支持 RHA；
- **reversal takeover**：反向接管更强，支持 RHA_REVERSAL；
- **active dominance**：低/中性清算压力下价格主动突破，支持 VT；
- **quiet no pressure**：压力弱、价格安静，支持 RC；
- **mixed pressure chop**：压力混杂且价格震荡，支持 AMB。

### 对六状态模型的作用

path multiscale 是六状态语义的核心底盘：HPEM/RHA 的区别、VT/RHA 的区别、RC/AMB 的区别，都高度依赖 path context 与 path label。

### 与其他特征层的关系

PLIE 给当前压力；path 给多尺度路径；price context 验证价格行为是否符合状态定义；absorption memory 给已成熟短期裁决。

### 缺失或错误的风险

缺失 multiscale path 会导致旧压力无法退出、RC 被压缩、RHA 过宽、VT/RHA 混淆。path 中若包含未来窗口响应，会直接造成泄漏。

### 因果与工程约束

- 所有 path features 必须基于过去窗口；
- `available_time <= time`；
- path_context 不应看未来价格；
- path_label 只能使用截至当前窗口已发生的价格路径。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `available_time` | 该特征在实盘中变为可用的时间。 | 防止把尚未成熟的 path 或 market response 当成当前证据。 | 由上游特征工程根据窗口结束时间或响应 horizon 计算；必须满足 available_time <= time。 |
| `path_context_6h` | 对应窗口的 path context，描述窗口内 PLIE / liquidation pressure 的结构。 | 回答压力背景是 directional、mixed 还是 neutral；是区分 RHA/HPEM/VT/RC 的关键条件。 | 由窗口内 PLIE direction、reliability、intensity 与 pressure mass/directionality 计算，不直接看未来价格。 |
| `path_context_12h` | 对应窗口的 path context，描述窗口内 PLIE / liquidation pressure 的结构。 | 回答压力背景是 directional、mixed 还是 neutral；是区分 RHA/HPEM/VT/RC 的关键条件。 | 由窗口内 PLIE direction、reliability、intensity 与 pressure mass/directionality 计算，不直接看未来价格。 |
| `path_context_24h` | 对应窗口的 path context，描述窗口内 PLIE / liquidation pressure 的结构。 | 回答压力背景是 directional、mixed 还是 neutral；是区分 RHA/HPEM/VT/RC 的关键条件。 | 由窗口内 PLIE direction、reliability、intensity 与 pressure mass/directionality 计算，不直接看未来价格。 |
| `path_context_48h` | 对应窗口的 path context，描述窗口内 PLIE / liquidation pressure 的结构。 | 回答压力背景是 directional、mixed 还是 neutral；是区分 RHA/HPEM/VT/RC 的关键条件。 | 由窗口内 PLIE direction、reliability、intensity 与 pressure mass/directionality 计算，不直接看未来价格。 |
| `path_label_6h` | 对应窗口的 path label，描述价格路径如何响应该窗口压力背景。 | 回答压力是被传导、级联、吸收、拒绝、主动突破还是安静无压力。 | 由窗口内截至 time 的 past-only 价格路径与 pressure context 对比生成。 |
| `path_label_12h` | 对应窗口的 path label，描述价格路径如何响应该窗口压力背景。 | 回答压力是被传导、级联、吸收、拒绝、主动突破还是安静无压力。 | 由窗口内截至 time 的 past-only 价格路径与 pressure context 对比生成。 |
| `path_label_24h` | 对应窗口的 path label，描述价格路径如何响应该窗口压力背景。 | 回答压力是被传导、级联、吸收、拒绝、主动突破还是安静无压力。 | 由窗口内截至 time 的 past-only 价格路径与 pressure context 对比生成。 |
| `path_label_48h` | 对应窗口的 path label，描述价格路径如何响应该窗口压力背景。 | 回答压力是被传导、级联、吸收、拒绝、主动突破还是安静无压力。 | 由窗口内截至 time 的 past-only 价格路径与 pressure context 对比生成。 |
| `path_absorption_score_6h` | 对应窗口的 path absorption 分数。 | 分数越高，说明强制流更可能被市场承接或钝化，支持 RHA/RC，削弱 HPEM。 | 由压力基线与窗口内已发生价格响应差异计算。 |
| `path_absorption_score_12h` | 对应窗口的 path absorption 分数。 | 分数越高，说明强制流更可能被市场承接或钝化，支持 RHA/RC，削弱 HPEM。 | 由压力基线与窗口内已发生价格响应差异计算。 |
| `path_absorption_score_24h` | 对应窗口的 path absorption 分数。 | 分数越高，说明强制流更可能被市场承接或钝化，支持 RHA/RC，削弱 HPEM。 | 由压力基线与窗口内已发生价格响应差异计算。 |
| `path_absorption_score_48h` | 对应窗口的 path absorption 分数。 | 分数越高，说明强制流更可能被市场承接或钝化，支持 RHA/RC，削弱 HPEM。 | 由压力基线与窗口内已发生价格响应差异计算。 |
| `path_pressure_rejection_score_6h` | 对应窗口的压力拒绝分数。 | 高值表示价格不服从 PLIE 压力，支持 RHA。 | 由压力方向与窗口收益/路径方向相反程度计算。 |
| `path_pressure_rejection_score_12h` | 对应窗口的压力拒绝分数。 | 高值表示价格不服从 PLIE 压力，支持 RHA。 | 由压力方向与窗口收益/路径方向相反程度计算。 |
| `path_pressure_rejection_score_24h` | 对应窗口的压力拒绝分数。 | 高值表示价格不服从 PLIE 压力，支持 RHA。 | 由压力方向与窗口收益/路径方向相反程度计算。 |
| `path_pressure_rejection_score_48h` | 对应窗口的压力拒绝分数。 | 高值表示价格不服从 PLIE 压力，支持 RHA。 | 由压力方向与窗口收益/路径方向相反程度计算。 |
| `path_active_dominance_score_6h` | 对应窗口主动交易主导分数。 | 高值支持 VT，尤其 pressure neutral/mixed 时。 | 由价格位移、趋势强度与低 PLIE 解释力组合计算。 |
| `path_active_dominance_score_12h` | 对应窗口主动交易主导分数。 | 高值支持 VT，尤其 pressure neutral/mixed 时。 | 由价格位移、趋势强度与低 PLIE 解释力组合计算。 |
| `path_active_dominance_score_24h` | 对应窗口主动交易主导分数。 | 高值支持 VT，尤其 pressure neutral/mixed 时。 | 由价格位移、趋势强度与低 PLIE 解释力组合计算。 |
| `path_active_dominance_score_48h` | 对应窗口主动交易主导分数。 | 高值支持 VT，尤其 pressure neutral/mixed 时。 | 由价格位移、趋势强度与低 PLIE 解释力组合计算。 |
| `path_active_dominance_price_score_6h` | 对应窗口主动主导价格方向分数。 | 表示在 PLIE 中性或混合时，价格主动向上/向下位移的强度与方向。 | 由 active dominance path label 与窗口收益方向/幅度计算。 |
| `path_active_dominance_price_score_12h` | 对应窗口主动主导价格方向分数。 | 表示在 PLIE 中性或混合时，价格主动向上/向下位移的强度与方向。 | 由 active dominance path label 与窗口收益方向/幅度计算。 |
| `path_active_dominance_price_score_24h` | 对应窗口主动主导价格方向分数。 | 表示在 PLIE 中性或混合时，价格主动向上/向下位移的强度与方向。 | 由 active dominance path label 与窗口收益方向/幅度计算。 |
| `path_active_dominance_price_score_48h` | 对应窗口主动主导价格方向分数。 | 表示在 PLIE 中性或混合时，价格主动向上/向下位移的强度与方向。 | 由 active dominance path label 与窗口收益方向/幅度计算。 |
| `path_transmission_ratio_6h` | 对应窗口清算压力传导比例。 | 衡量价格实际位移相对 PLIE/压力基线的传导程度；高值支持 HPEM/ST，负值支持 RHA。 | 由压力方向对齐收益除以压力基线幅度计算。 |
| `path_transmission_ratio_12h` | 对应窗口清算压力传导比例。 | 衡量价格实际位移相对 PLIE/压力基线的传导程度；高值支持 HPEM/ST，负值支持 RHA。 | 由压力方向对齐收益除以压力基线幅度计算。 |
| `path_transmission_ratio_24h` | 对应窗口清算压力传导比例。 | 衡量价格实际位移相对 PLIE/压力基线的传导程度；高值支持 HPEM/ST，负值支持 RHA。 | 由压力方向对齐收益除以压力基线幅度计算。 |
| `path_transmission_ratio_48h` | 对应窗口清算压力传导比例。 | 衡量价格实际位移相对 PLIE/压力基线的传导程度；高值支持 HPEM/ST，负值支持 RHA。 | 由压力方向对齐收益除以压力基线幅度计算。 |
| `path_direction_consistency_6h` | 对应窗口方向一致性。 | 衡量窗口内压力或价格响应方向是否持续一致。 | 由窗口内方向符号一致比例或相关 score 计算。 |
| `path_direction_consistency_12h` | 对应窗口方向一致性。 | 衡量窗口内压力或价格响应方向是否持续一致。 | 由窗口内方向符号一致比例或相关 score 计算。 |
| `path_direction_consistency_24h` | 对应窗口方向一致性。 | 衡量窗口内压力或价格响应方向是否持续一致。 | 由窗口内方向符号一致比例或相关 score 计算。 |
| `path_direction_consistency_48h` | 对应窗口方向一致性。 | 衡量窗口内压力或价格响应方向是否持续一致。 | 由窗口内方向符号一致比例或相关 score 计算。 |
| `path_cascade_score_6h` | 对应窗口级联传导分数。 | 高值表示压力持续穿透并放大，支持 HPEM。 | 由 cascade label、传导比例、波动/跳跃等窗口证据计算。 |
| `path_cascade_score_12h` | 对应窗口级联传导分数。 | 高值表示压力持续穿透并放大，支持 HPEM。 | 由 cascade label、传导比例、波动/跳跃等窗口证据计算。 |
| `path_cascade_score_24h` | 对应窗口级联传导分数。 | 高值表示压力持续穿透并放大，支持 HPEM。 | 由 cascade label、传导比例、波动/跳跃等窗口证据计算。 |
| `path_cascade_score_48h` | 对应窗口级联传导分数。 | 高值表示压力持续穿透并放大，支持 HPEM。 | 由 cascade label、传导比例、波动/跳跃等窗口证据计算。 |
| `path_data_quality_6h` | 对应窗口 path 数据质量。 | 数据质量差时不应把 path evidence 作为高置信交易依据，可能推高 AMB。 | 由窗口缺失、时间间隔、价格/PLIE 可用性等计算。 |
| `path_data_quality_12h` | 对应窗口 path 数据质量。 | 数据质量差时不应把 path evidence 作为高置信交易依据，可能推高 AMB。 | 由窗口缺失、时间间隔、价格/PLIE 可用性等计算。 |
| `path_data_quality_24h` | 对应窗口 path 数据质量。 | 数据质量差时不应把 path evidence 作为高置信交易依据，可能推高 AMB。 | 由窗口缺失、时间间隔、价格/PLIE 可用性等计算。 |
| `path_data_quality_48h` | 对应窗口 path 数据质量。 | 数据质量差时不应把 path evidence 作为高置信交易依据，可能推高 AMB。 | 由窗口缺失、时间间隔、价格/PLIE 可用性等计算。 |
| `path_signal_clarity_6h` | 对应窗口 path 信号清晰度。 | 信号清晰度低表示多个 path 解释竞争，可能提高 AMB 或降低 belief gap。 | 由 label margin、context/response 分歧等计算。 |
| `path_signal_clarity_12h` | 对应窗口 path 信号清晰度。 | 信号清晰度低表示多个 path 解释竞争，可能提高 AMB 或降低 belief gap。 | 由 label margin、context/response 分歧等计算。 |
| `path_signal_clarity_24h` | 对应窗口 path 信号清晰度。 | 信号清晰度低表示多个 path 解释竞争，可能提高 AMB 或降低 belief gap。 | 由 label margin、context/response 分歧等计算。 |
| `path_signal_clarity_48h` | 对应窗口 path 信号清晰度。 | 信号清晰度低表示多个 path 解释竞争，可能提高 AMB 或降低 belief gap。 | 由 label margin、context/response 分歧等计算。 |
| `path_activity_level_6h` | 对应窗口市场活动水平。 | 低 activity 可能支持 RC，但不等于数据质量差。 | 由 pressure mass、价格波动、趋势强度等活动量指标计算。 |
| `path_activity_level_12h` | 对应窗口市场活动水平。 | 低 activity 可能支持 RC，但不等于数据质量差。 | 由 pressure mass、价格波动、趋势强度等活动量指标计算。 |
| `path_activity_level_24h` | 对应窗口市场活动水平。 | 低 activity 可能支持 RC，但不等于数据质量差。 | 由 pressure mass、价格波动、趋势强度等活动量指标计算。 |
| `path_activity_level_48h` | 对应窗口市场活动水平。 | 低 activity 可能支持 RC，但不等于数据质量差。 | 由 pressure mass、价格波动、趋势强度等活动量指标计算。 |

---

## 10. `path_absorption.csv`：24h 兼容层与审计字段

### 文件角色

`path_absorption.csv` 保留旧版 24h path 层，用于兼容、审计和 cross-check。它不能替代 multiscale path。

### 它回答的市场问题

旧版 24h path 对当前压力—响应结构如何判断？新版 multiscale path 与旧 24h path 是否一致？若冲突，是否需要审查上游特征或窗口逻辑？

### 核心金融含义

24h 是重要 episode horizon，但不是唯一 horizon。保留 legacy 24h 层可以帮助追溯历史版本、比较模型迭代前后的状态语义。

### 对六状态模型的作用

主要作为 audit / compatibility layer；主状态语义应优先来自 multiscale path。

### 缺失或错误的风险

无法进行旧版 path 兼容审计，历史结果和新版结果可比性下降。

### 因果与工程约束

`available_time <= time`；不能让旧版字段覆盖新版 multiscale path 主逻辑。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `available_time` | 该特征在实盘中变为可用的时间。 | 防止把尚未成熟的 path 或 market response 当成当前证据。 | 由上游特征工程根据窗口结束时间或响应 horizon 计算；必须满足 available_time <= time。 |
| `window_hours` | path_absorption.csv 的窗口长度。 | 标识该 legacy path 记录对应 24h 或其他窗口；当前作为兼容/审计层使用。 | 由上游 path 模块写入。 |
| `path_context` | 对应窗口的 path context，描述窗口内 PLIE / liquidation pressure 的结构。 | 回答压力背景是 directional、mixed 还是 neutral；是区分 RHA/HPEM/VT/RC 的关键条件。 | 由窗口内 PLIE direction、reliability、intensity 与 pressure mass/directionality 计算，不直接看未来价格。 |
| `path_label` | 对应窗口的 path label，描述价格路径如何响应该窗口压力背景。 | 回答压力是被传导、级联、吸收、拒绝、主动突破还是安静无压力。 | 由窗口内截至 time 的 past-only 价格路径与 pressure context 对比生成。 |
| `path_absorption_score_0_100` | 对应窗口的 path absorption 分数。 | 分数越高，说明强制流更可能被市场承接或钝化，支持 RHA/RC，削弱 HPEM。 | 由压力基线与窗口内已发生价格响应差异计算。 |
| `path_pressure_rejection_score` | 对应窗口的压力拒绝分数。 | 高值表示价格不服从 PLIE 压力，支持 RHA。 | 由压力方向与窗口收益/路径方向相反程度计算。 |
| `path_active_dominance_score` | 对应窗口主动交易主导分数。 | 高值支持 VT，尤其 pressure neutral/mixed 时。 | 由价格位移、趋势强度与低 PLIE 解释力组合计算。 |
| `path_active_dominance_price_score` | 对应窗口主动主导价格方向分数。 | 表示在 PLIE 中性或混合时，价格主动向上/向下位移的强度与方向。 | 由 active dominance path label 与窗口收益方向/幅度计算。 |
| `path_transmission_ratio` | 对应窗口清算压力传导比例。 | 衡量价格实际位移相对 PLIE/压力基线的传导程度；高值支持 HPEM/ST，负值支持 RHA。 | 由压力方向对齐收益除以压力基线幅度计算。 |
| `path_direction_consistency` | 对应窗口方向一致性。 | 衡量窗口内压力或价格响应方向是否持续一致。 | 由窗口内方向符号一致比例或相关 score 计算。 |
| `path_quality` | legacy path quality。 | 旧版综合质量指标；新版更推荐拆分 data_quality / signal_clarity / activity_level。 | 由旧版 path_absorption 模块计算。 |
| `path_cascade_score` | 对应窗口级联传导分数。 | 高值表示压力持续穿透并放大，支持 HPEM。 | 由 cascade label、传导比例、波动/跳跃等窗口证据计算。 |

---

## 11. `price_context_features.csv`：价格行为上下文

### 文件角色

`price_context_features.csv` 是 full price-context design 的核心文件。它描述当前已发生价格行为：波动、区间压缩、趋势强度、一致性、跳跃、价格质量等。

### 它回答的市场问题

当前价格行为是否符合某个状态的金融定义？RC 是否真低波低趋势？VT 是否真有主动位移？HPEM 是否真有压力穿透？RHA 是否体现压力拒绝或反向接管？ST 是否温和有序？

### 核心金融含义

price context 不是为了直接预测未来收益。它用于裁决“状态是否仍成立”。例如，高 PLIE 只是压力源，价格是否顺 PLIE 穿透要看 realized vol、jump、trend 和 direction；RC 不能只靠 quiet path，还需要低波动、压缩、低趋势；VT 需要主动位移和较大价格变化。

### 对六状态模型的作用

- RC：低 vol、high range compression、low trend；
- VT：高 vol / active move / trend change；
- ST：温和、有序、非极端趋势；
- HPEM：清算压力与价格冲击共振；
- RHA：价格是否拒绝 PLIE 或反向接管；
- AMB：数据质量异常或价格/路径冲突。

### 与其他特征层的关系

price context 服务于 semantic evidence、directional quality、state confirmation、price expectation compliance 和 segment diagnostics。

### 缺失或错误的风险

若缺失，RC/VT/ST/HPEM/RHA 的价格行为边界会变模糊，低质量状态段可能被误用作交易触发。

### 因果与工程约束

所有窗口特征必须 past-only；`price_feature_time <= time`；只能 backward-asof join；price gap/missing/outlier 必须显式标记。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `price_feature_time` | price context 实际计算使用的最新价格特征时间。 | 用于检查 price context 是否滞后。 | 由 price_context_features 上游写入。 |
| `price_feature_age_min` | price context 相对当前 time 的年龄，单位分钟。 | 年龄过大表示价格上下文 stale，应降权或触发质量风险。 | time - price_feature_time。 |
| `past_return_1h_bps` | 对应窗口的 past-only 对数收益，单位 bps。 | 描述截至当前的价格位移，不是未来收益预测；用于判断 VT/ST/HPEM/RHA 的价格行为是否成立。 | 10000 * log(P_t / P_{t-W})。 |
| `past_return_3h_bps` | 对应窗口的 past-only 对数收益，单位 bps。 | 描述截至当前的价格位移，不是未来收益预测；用于判断 VT/ST/HPEM/RHA 的价格行为是否成立。 | 10000 * log(P_t / P_{t-W})。 |
| `past_return_6h_bps` | 对应窗口的 past-only 对数收益，单位 bps。 | 描述截至当前的价格位移，不是未来收益预测；用于判断 VT/ST/HPEM/RHA 的价格行为是否成立。 | 10000 * log(P_t / P_{t-W})。 |
| `past_return_12h_bps` | 对应窗口的 past-only 对数收益，单位 bps。 | 描述截至当前的价格位移，不是未来收益预测；用于判断 VT/ST/HPEM/RHA 的价格行为是否成立。 | 10000 * log(P_t / P_{t-W})。 |
| `past_return_24h_bps` | 对应窗口的 past-only 对数收益，单位 bps。 | 描述截至当前的价格位移，不是未来收益预测；用于判断 VT/ST/HPEM/RHA 的价格行为是否成立。 | 10000 * log(P_t / P_{t-W})。 |
| `realized_vol_1h_bps` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_6h_bps` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_24h_bps` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_1h_per_sqrt_hour_bps` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_6h_per_sqrt_hour_bps` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_24h_per_sqrt_hour_bps` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_1h_z` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_6h_z` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `realized_vol_24h_z` | 对应窗口 realized volatility。 | 衡量已发生价格风险释放；RC 应低，VT/HPEM 通常较高。 | 窗口内过去收益平方和开方，或按小时归一化/robust z-score。 |
| `range_width_1h_bps` | 对应窗口最高价与最低价形成的区间宽度，单位 bps。 | 衡量价格是否展开区间；RC 应较窄，VT/HPEM 可能较宽。 | 10000 * log(max price / min price) 或 OHLC high-low。 |
| `range_width_6h_bps` | 对应窗口最高价与最低价形成的区间宽度，单位 bps。 | 衡量价格是否展开区间；RC 应较窄，VT/HPEM 可能较宽。 | 10000 * log(max price / min price) 或 OHLC high-low。 |
| `range_width_24h_bps` | 对应窗口最高价与最低价形成的区间宽度，单位 bps。 | 衡量价格是否展开区间；RC 应较窄，VT/HPEM 可能较宽。 | 10000 * log(max price / min price) 或 OHLC high-low。 |
| `range_compression_1h` | 对应窗口区间压缩分数。 | 高值支持 RC 或低活动环境；低值表示 range expansion。 | 通常为 1 - past-only range percentile。 |
| `range_compression_6h` | 对应窗口区间压缩分数。 | 高值支持 RC 或低活动环境；低值表示 range expansion。 | 通常为 1 - past-only range percentile。 |
| `range_compression_24h` | 对应窗口区间压缩分数。 | 高值支持 RC 或低活动环境；低值表示 range expansion。 | 通常为 1 - past-only range percentile。 |
| `range_to_vol_1h` | 区间宽度与 realized vol 的比值。 | 辅助区分单边展开、震荡扫动与压缩。 | range_width / (realized_vol + epsilon)。 |
| `range_to_vol_6h` | 区间宽度与 realized vol 的比值。 | 辅助区分单边展开、震荡扫动与压缩。 | range_width / (realized_vol + epsilon)。 |
| `range_to_vol_24h` | 区间宽度与 realized vol 的比值。 | 辅助区分单边展开、震荡扫动与压缩。 | range_width / (realized_vol + epsilon)。 |
| `trend_efficiency_1h` | 趋势效率。 | 衡量净位移相对路径长度的效率；高值表示路径更单边。 | abs(window return) / (sum(abs(bar returns)) + epsilon)。 |
| `trend_efficiency_6h` | 趋势效率。 | 衡量净位移相对路径长度的效率；高值表示路径更单边。 | abs(window return) / (sum(abs(bar returns)) + epsilon)。 |
| `trend_efficiency_24h` | 趋势效率。 | 衡量净位移相对路径长度的效率；高值表示路径更单边。 | abs(window return) / (sum(abs(bar returns)) + epsilon)。 |
| `trend_snr_1h` | 趋势信噪比。 | 衡量窗口净位移相对波动的显著性。 | abs(window return) / (realized_vol + epsilon)。 |
| `trend_snr_6h` | 趋势信噪比。 | 衡量窗口净位移相对波动的显著性。 | abs(window return) / (realized_vol + epsilon)。 |
| `trend_snr_24h` | 趋势信噪比。 | 衡量窗口净位移相对波动的显著性。 | abs(window return) / (realized_vol + epsilon)。 |
| `trend_slope_1h` | log price 线性趋势斜率。 | 衡量窗口内价格路径的方向与速度。 | 对窗口内 log price 回归时间索引得到斜率。 |
| `trend_slope_6h` | log price 线性趋势斜率。 | 衡量窗口内价格路径的方向与速度。 | 对窗口内 log price 回归时间索引得到斜率。 |
| `trend_slope_24h` | log price 线性趋势斜率。 | 衡量窗口内价格路径的方向与速度。 | 对窗口内 log price 回归时间索引得到斜率。 |
| `trend_slope_tstat_1h` | log price 线性趋势斜率的 t 统计量。 | 衡量趋势是否有统计上稳定的方向性。 | 对窗口内 log price 回归时间索引，取斜率 t-stat。 |
| `trend_slope_tstat_6h` | log price 线性趋势斜率的 t 统计量。 | 衡量趋势是否有统计上稳定的方向性。 | 对窗口内 log price 回归时间索引，取斜率 t-stat。 |
| `trend_slope_tstat_24h` | log price 线性趋势斜率的 t 统计量。 | 衡量趋势是否有统计上稳定的方向性。 | 对窗口内 log price 回归时间索引，取斜率 t-stat。 |
| `trend_r2_1h` | 趋势回归 R²。 | 衡量价格是否沿线性趋势有序推进；ST 更需要较高一致性。 | 窗口内 log price 线性回归的 R²。 |
| `trend_r2_6h` | 趋势回归 R²。 | 衡量价格是否沿线性趋势有序推进；ST 更需要较高一致性。 | 窗口内 log price 线性回归的 R²。 |
| `trend_r2_24h` | 趋势回归 R²。 | 衡量价格是否沿线性趋势有序推进；ST 更需要较高一致性。 | 窗口内 log price 线性回归的 R²。 |
| `trend_strength_1h` | 趋势强度综合分数。 | VT/ST/HPEM 需要不同程度的趋势强度；RC 需要低趋势。 | 由 trend efficiency、trend SNR、slope t-stat 等组合。 |
| `trend_strength_6h` | 趋势强度综合分数。 | VT/ST/HPEM 需要不同程度的趋势强度；RC 需要低趋势。 | 由 trend efficiency、trend SNR、slope t-stat 等组合。 |
| `trend_strength_24h` | 趋势强度综合分数。 | VT/ST/HPEM 需要不同程度的趋势强度；RC 需要低趋势。 | 由 trend efficiency、trend SNR、slope t-stat 等组合。 |
| `trend_direction_1h` | 趋势方向。 | 用于 directional_state、VT_UP/DOWN、HPEM/RHA price expectation。 | 通常为 sign(past_return_W) 或趋势斜率方向。 |
| `trend_direction_6h` | 趋势方向。 | 用于 directional_state、VT_UP/DOWN、HPEM/RHA price expectation。 | 通常为 sign(past_return_W) 或趋势斜率方向。 |
| `trend_direction_24h` | 趋势方向。 | 用于 directional_state、VT_UP/DOWN、HPEM/RHA price expectation。 | 通常为 sign(past_return_W) 或趋势斜率方向。 |
| `bar_direction_align_1h` | 窗口内单根 bar 与窗口净方向一致的比例。 | 衡量趋势是否连续，而不是单根跳跃。 | 统计 sign(r_i) 与 sign(R_W) 一致的比例。 |
| `bar_direction_align_6h` | 窗口内单根 bar 与窗口净方向一致的比例。 | 衡量趋势是否连续，而不是单根跳跃。 | 统计 sign(r_i) 与 sign(R_W) 一致的比例。 |
| `bar_direction_align_24h` | 窗口内单根 bar 与窗口净方向一致的比例。 | 衡量趋势是否连续，而不是单根跳跃。 | 统计 sign(r_i) 与 sign(R_W) 一致的比例。 |
| `block_direction_align_6h` | 窗口内子窗口方向一致比例。 | 衡量趋势在多个子区间是否一致。 | 将窗口拆成 block，统计 block return 方向与总体方向一致比例。 |
| `block_direction_align_24h` | 窗口内子窗口方向一致比例。 | 衡量趋势在多个子区间是否一致。 | 将窗口拆成 block，统计 block return 方向与总体方向一致比例。 |
| `trend_consistency_1h` | 趋势一致性综合分数。 | ST/HPEM/RHA 方向生命周期、RC 排除、VT 方向层的重要证据。 | 由 bar align、block align、trend R² 等组合。 |
| `trend_consistency_6h` | 趋势一致性综合分数。 | ST/HPEM/RHA 方向生命周期、RC 排除、VT 方向层的重要证据。 | 由 bar align、block align、trend R² 等组合。 |
| `trend_consistency_24h` | 趋势一致性综合分数。 | ST/HPEM/RHA 方向生命周期、RC 排除、VT 方向层的重要证据。 | 由 bar align、block align、trend R² 等组合。 |
| `vol_of_vol_6h` | 对应窗口 volatility-of-volatility。 | 高值表示波动结构不稳定，可能支持 VT/HPEM/AMB 或切换期。 | 窗口内 1h realized vol 的变异系数。 |
| `vol_of_vol_24h` | 对应窗口 volatility-of-volatility。 | 高值表示波动结构不稳定，可能支持 VT/HPEM/AMB 或切换期。 | 窗口内 1h realized vol 的变异系数。 |
| `vol_of_vol_48h` | 对应窗口 volatility-of-volatility。 | 高值表示波动结构不稳定，可能支持 VT/HPEM/AMB 或切换期。 | 窗口内 1h realized vol 的变异系数。 |
| `vol_of_vol_abs_6h` | 对应窗口 volatility-of-volatility 的绝对形式。 | 衡量波动率自身是否剧烈变化，帮助识别状态迁移期。 | 窗口内 1h realized vol 的标准差。 |
| `vol_of_vol_abs_24h` | 对应窗口 volatility-of-volatility 的绝对形式。 | 衡量波动率自身是否剧烈变化，帮助识别状态迁移期。 | 窗口内 1h realized vol 的标准差。 |
| `vol_of_vol_abs_48h` | 对应窗口 volatility-of-volatility 的绝对形式。 | 衡量波动率自身是否剧烈变化，帮助识别状态迁移期。 | 窗口内 1h realized vol 的标准差。 |
| `max_jump_z_1h` | 窗口内最大单 bar jump z-score。 | 识别离散冲击；HPEM/VT 价格冲击确认的重要证据。 | abs(return) / robust rolling return scale 的窗口最大值。 |
| `max_jump_z_6h` | 窗口内最大单 bar jump z-score。 | 识别离散冲击；HPEM/VT 价格冲击确认的重要证据。 | abs(return) / robust rolling return scale 的窗口最大值。 |
| `max_jump_z_24h` | 窗口内最大单 bar jump z-score。 | 识别离散冲击；HPEM/VT 价格冲击确认的重要证据。 | abs(return) / robust rolling return scale 的窗口最大值。 |
| `jump_count_1h` | 窗口内超过阈值的 jump 数量。 | 衡量冲击是否频繁；辅助判断极端运动或噪声风险。 | 统计 JumpZ > threshold 的 bar 数。 |
| `jump_count_6h` | 窗口内超过阈值的 jump 数量。 | 衡量冲击是否频繁；辅助判断极端运动或噪声风险。 | 统计 JumpZ > threshold 的 bar 数。 |
| `jump_count_24h` | 窗口内超过阈值的 jump 数量。 | 衡量冲击是否频繁；辅助判断极端运动或噪声风险。 | 统计 JumpZ > threshold 的 bar 数。 |
| `jump_ratio_bv_6h` | 基于 bipower variation 的跳跃占比。 | 区分连续波动与离散跳跃冲击。 | max(RV - BV, 0) / (RV + epsilon)。 |
| `jump_ratio_bv_24h` | 基于 bipower variation 的跳跃占比。 | 区分连续波动与离散跳跃冲击。 | max(RV - BV, 0) / (RV + epsilon)。 |
| `jump_proxy_1h` | 跳跃冲击综合分数。 | HPEM / VT / price shock fast lane 的关键价格证据。 | 由 max_jump_z、jump_ratio、jump_count 等组合。 |
| `jump_proxy_6h` | 跳跃冲击综合分数。 | HPEM / VT / price shock fast lane 的关键价格证据。 | 由 max_jump_z、jump_ratio、jump_count 等组合。 |
| `jump_proxy_24h` | 跳跃冲击综合分数。 | HPEM / VT / price shock fast lane 的关键价格证据。 | 由 max_jump_z、jump_ratio、jump_count 等组合。 |
| `signed_max_jump_return_1h_bps` | 窗口内绝对值最大跳跃的有符号收益，单位 bps。 | 给出最大冲击方向，用于方向解释。 | 取 abs(return) 最大的 bar return，并保留符号。 |
| `signed_max_jump_return_6h_bps` | 窗口内绝对值最大跳跃的有符号收益，单位 bps。 | 给出最大冲击方向，用于方向解释。 | 取 abs(return) 最大的 bar return，并保留符号。 |
| `signed_max_jump_return_24h_bps` | 窗口内绝对值最大跳跃的有符号收益，单位 bps。 | 给出最大冲击方向，用于方向解释。 | 取 abs(return) 最大的 bar return，并保留符号。 |
| `price_missing_ratio_1h` | 对应窗口价格缺失比例。 | 缺失过高时 price context 不可靠，可能提高 AMB 或 fail quality。 | 窗口内缺失观测数 / 期望观测数。 |
| `price_missing_ratio_6h` | 对应窗口价格缺失比例。 | 缺失过高时 price context 不可靠，可能提高 AMB 或 fail quality。 | 窗口内缺失观测数 / 期望观测数。 |
| `price_missing_ratio_24h` | 对应窗口价格缺失比例。 | 缺失过高时 price context 不可靠，可能提高 AMB 或 fail quality。 | 窗口内缺失观测数 / 期望观测数。 |
| `price_gap_flag_1h` | 对应窗口价格时间间隔或数据 gap 标记。 | 防止把数据断档误读为价格冲击。 | 由时间间隔异常、缺失连续性等规则生成。 |
| `price_gap_flag_6h` | 对应窗口价格时间间隔或数据 gap 标记。 | 防止把数据断档误读为价格冲击。 | 由时间间隔异常、缺失连续性等规则生成。 |
| `price_gap_flag_24h` | 对应窗口价格时间间隔或数据 gap 标记。 | 防止把数据断档误读为价格冲击。 | 由时间间隔异常、缺失连续性等规则生成。 |
| `price_outlier_flag_1h` | 对应窗口价格极端异常标记。 | 区分真实极端行情与数据异常，需要结合 gap/missing 审查。 | 由 return z-score 或鲁棒阈值检测。 |
| `price_outlier_flag_6h` | 对应窗口价格极端异常标记。 | 区分真实极端行情与数据异常，需要结合 gap/missing 审查。 | 由 return z-score 或鲁棒阈值检测。 |
| `price_outlier_flag_24h` | 对应窗口价格极端异常标记。 | 区分真实极端行情与数据异常，需要结合 gap/missing 审查。 | 由 return z-score 或鲁棒阈值检测。 |

---

## 12. `liqprice_features.csv`：清算价格微观上下文辅助层

### 文件角色

`liqprice_features.csv` 是 legacy price/liquidation micro-context auxiliary channel。它是 required input，但属于辅助通道。

### 它回答的市场问题

旧版价格—清算微观 proxy 是否支持当前压力背景？清算价格相关 spike、velocity、acceleration 是否存在异常？

### 核心金融含义

该文件有审计和旧版兼容价值，但它不是六状态主裁决层。required 不等于 dominant。

### 对六状态模型的作用

辅助诊断价格—清算微观关系；在 dashboard、audit、旧版本对齐中有价值。

### 与其他特征层的关系

不能压过 PLIE/HMM、matured memory、multiscale path、price context 和 quality/conflict layer。

### 缺失或错误的风险

feature universe 不一致，旧版对齐和辅助审计能力下降。

### 因果与工程约束

必须 backward-asof 到 source-clock；不能让 auxiliary channel 覆盖主语义层。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `trend_pressure` | legacy price/liquidation trend pressure。 | 辅助观察价格与清算压力的微观关系，但不应主导六状态。 | 由旧版 liqprice 模块计算。 |
| `kalman_slope` | Kalman 平滑后的价格/趋势斜率。 | 辅助趋势背景审计，尤其旧版兼容。 | 由上游 Kalman filter 或状态空间平滑估计。 |
| `vol_adaptive` | 自适应波动 proxy。 | 辅助判断波动环境，但新版主 price context 已更完整。 | 由旧版价格波动模块计算。 |
| `fll_spike_kama` | long/short liquidation price micro-context 特征。 | 辅助诊断多空清算压力的 spike、velocity、acceleration。 | 由 liqprice 特征工程基于清算价格/压力序列计算。 |
| `fsl_spike_kama` | long/short liquidation price micro-context 特征。 | 辅助诊断多空清算压力的 spike、velocity、acceleration。 | 由 liqprice 特征工程基于清算价格/压力序列计算。 |
| `fll_velocity_gaussian` | long/short liquidation price micro-context 特征。 | 辅助诊断多空清算压力的 spike、velocity、acceleration。 | 由 liqprice 特征工程基于清算价格/压力序列计算。 |
| `fsl_velocity_gaussian` | long/short liquidation price micro-context 特征。 | 辅助诊断多空清算压力的 spike、velocity、acceleration。 | 由 liqprice 特征工程基于清算价格/压力序列计算。 |
| `fll_acceleration_gaussian` | long/short liquidation price micro-context 特征。 | 辅助诊断多空清算压力的 spike、velocity、acceleration。 | 由 liqprice 特征工程基于清算价格/压力序列计算。 |
| `fsl_acceleration_gaussian` | long/short liquidation price micro-context 特征。 | 辅助诊断多空清算压力的 spike、velocity、acceleration。 | 由 liqprice 特征工程基于清算价格/压力序列计算。 |

---

## 13. `features_liq_dataflow.csv`：FHMV / liquidation-stress 辅助诊断

### 文件角色

`features_liq_dataflow.csv` 提供 FHMV / liquidation-stress 相关辅助诊断。它是 required input，用于保持 feature universe 固定。

### 它回答的市场问题

清算压力或波动压力背景是否存在异常？多空 stress 的波形或 dominance 是否与主语义层一致？

### 核心金融含义

FHMV 适合作为 volatility/stress submodule，不适合作为六状态主模型。六状态核心不是 volatility regime，而是清算压力、价格响应、吸收/传导/拒绝、主动交易和证据冲突。

### 对六状态模型的作用

辅助压力背景、异常诊断、旧特征一致性审计。不能直接主导 HPEM/RHA/VT 等状态。

### 与其他特征层的关系

若与主语义层冲突，应由 quality/conflict 或诊断逻辑处理，而不是让 FHMV 覆盖主状态判断。

### 缺失或错误的风险

压力辅助诊断缺失，版本可比性下降；但即便存在，也不能单独决定六状态。

### 因果与工程约束

必须对齐到 source-clock；作为 auxiliary channel 进入，不允许变成 dominant state evidence。

### 核心列说明

| 核心列 | 简要描述 | 金融学意义 | 计算方法 / 来源 |
|---|---|---|---|
| `time` | 记录行对应的模型推理时点。 | 定义 live inference 在哪个时点生成状态。所有特征必须在该时点或之前可得。 | 由上游数据时钟产生；在 canonical frame 中作为 join key。 |
| `z_logtotalp` | 总清算压力或总压力规模的 z-score/log 形式。 | 辅助衡量 liquidation-stress 背景。 | 由 FHMV/liq stress 模块对总压力取 log 并标准化。 |
| `z_sdom` | side dominance 的标准化指标。 | 辅助观察多空压力主导程度。 | 由 FHMV/liq stress 模块计算并标准化。 |
| `risk_priority_number` | FHMV/liq stress 模块输出的压力或风险 proxy。 | 辅助诊断压力异常，不直接主导六状态。 | 由上游 FHMV 特征工程生成。 |
| `dominance` | FHMV/liq stress 模块的 dominance 指标。 | 辅助诊断多空压力主导背景。 | 由上游 FHMV 特征工程生成。 |
| `z_fll_cwt_kf` | long/short liquidation stress wavelet/Kalman 标准化特征。 | 辅助诊断清算压力波形和多空 stress 背景。 | 由 FHMV 模块对 liquidation-side features 做滤波/标准化。 |
| `z_fsl_cwt_kf` | long/short liquidation stress wavelet/Kalman 标准化特征。 | 辅助诊断清算压力波形和多空 stress 背景。 | 由 FHMV 模块对 liquidation-side features 做滤波/标准化。 |

---

## 14. 主状态语义来自哪些层

六状态主语义来自以下组合，而不是任何单一文件：

```text
PLIE / HMM
+ matured market response memory
+ multiscale path absorption
+ price context
+ quality / conflict layer
+ duration and transition constraints
```

这意味着：

- 高 PLIE 不自动等于 HPEM；
- 高 absorption 不自动等于可交易反转；
- 高波动不自动等于 VT 或 HPEM；
- low activity 不自动等于 AMB；
- auxiliary channel 不允许覆盖主语义层。

---

## 15. No-leakage 与因果对齐原则

### source-clock

`source-clock` 是模型的主推理时钟。当前模型以 `base_context.csv.time` 为主时钟。所有输入都必须对齐到该时钟。

### feature_time <= t

任何特征的实际生成时间必须不晚于当前推理时点。若 `feature_time > t`，则说明模型偷看了未来。

### backward-asof join

异步特征必须使用 backward-asof join：只向后寻找最近一个已经可得的特征，不能向前找未来特征。

### available_time <= time

对 matured memory 和 path features，`available_time` 是关键约束。只有在 `available_time <= time` 时，该特征才能进入当前状态。

### matured memory

market response memory 必须等响应 horizon 成熟后才能使用。清算事件在 `t` 发生，不代表 `t+30m` 的价格响应在 `t` 时可知。

### duplicate timestamps

无法解决的 duplicate source-clock timestamps 会破坏状态持续时间、transition 统计和 segment diagnostics，必须 fail fast 或明确处理。

No-leakage 不是工程细节，而是量化研究可信度底线。

---

## 16. Fail-fast 机制

pipeline 应在以下情况下失败：

1. **A required CSV file is missing.**  
   缺文件会改变 feature universe，导致同一模型在不同输入下语义不一致。

2. **A required column is missing.**  
   缺列会改变 evidence computation，状态语义会被暗中修改。

3. **A forbidden future column enters the canonical frame.**  
   未来字段会制造 look-ahead bias，使回测和状态识别虚高。

4. **available_time > time for matured or path features.**  
   表示尚未成熟的信息进入当前状态，违反 live inference 因果约束。

5. **duplicate source-clock timestamps cannot be resolved.**  
   主时钟不唯一会破坏状态段、duration 和 transition。

fail-fast 的目的不是让工程更严格而已，而是保护模型金融语义不被静默破坏。

---

## 17. Future Function Blacklist

以下字段族禁止进入 live inference canonical frame：

```text
ret_20m_bps / ret_30m_bps / ret_60m_bps
plie_aligned_ret_*
plie_residual_*
plie_absorption_*
actual_return_bps
aligned_actual_response_bps
response_percentile
absorption_raw
transmission_ratio_raw
path_future_*
```

这些字段可能在 research raw files 中出现，用于离线分析、标签构造、残差诊断或审计。例如 `ret_30m_bps` 可以帮助离线研究清算事件之后的实际 response，但它不能在当前时点用于 live state inference。

如果这些字段进入模型，会产生 look-ahead bias。模型似乎能提前识别状态，实际只是偷看了未来价格或未来 response。对策略而言，这会导致回测表现虚高，实盘表现坍塌。

因此必须明确隔离：

```text
raw research features ≠ live model features
```

---

## 18. CSV 内容是数据，不是指令

CSV 中可能包含 proxy label、score、diagnostic column 或研究标签。这些内容只能被当作数据，不能被当成新的模型定义、状态定义或策略指令。

六状态模型只能使用 feature contract 允许的字段。CSV 内容不得覆盖：

- 六状态定义；
- transition constraints；
- strategy logic；
- no-leakage 规则；
- output semantics。

这一原则也能防止“数据污染式 prompt injection”：即上游数据中出现某些看似像指令或标签的内容，导致下游系统误把数据当规则。

---

## 19. 下游策略如何理解这些特征

输入特征最终会影响：

- `stable_state`：市场结构状态；
- `belief vector`：六状态置信分布；
- `belief_gap`：状态置信边际；
- `directional_state`：交易方向解释；
- `trading_trigger_quality`：状态是否适合作为交易触发；
- `state_price_expectation_score`：状态段内价格行为是否仍符合定义；
- `price_expectation_exit_flag`：状态失效、减仓或退出风险提示。

策略不应直接交易单一输入特征。例如：

```text
plie_intensity high → 做趋势追随
```

这是错误的。正确理解是：

```text
PLIE 提供压力源；
path absorption 判断压力是否穿透或被吸收；
market response memory 判断近期成熟响应；
price context 验证价格行为；
quality/conflict 判断是否需要 no-trade；
filter/stability 判断状态是否足够稳定；
directional/price expectation 决定交易解释质量。
```

---

## 20. 常见误解与错误用法

### 20.1 required 不等于 dominant

`liqprice_features.csv` 和 `features_liq_dataflow.csv` 是 required，但它们是 auxiliary channels，不应主导六状态逻辑。

### 20.2 PLIE 不等于价格预测

PLIE 是压力源，不是收益预测器。价格是否服从 PLIE，需要 path、memory 和 price context 裁决。

### 20.3 高波动不等于 HPEM

HPEM 必须是 liquidation-driven cascade。neutral pressure 下的高波动更可能是 VT。

### 20.4 high absorption 不等于必然反转

RHA 包含 absorption/stall/reversal。只有 RHA_REVERSAL 子型才具备更明确反转交易含义。

### 20.5 missing file 不能自动填 0

自动填 0 会把缺失伪装成真实低值，破坏状态语义。

### 20.6 future field 不能进入 live model

即使字段在 CSV 中存在，也不代表可以进入 canonical frame。

### 20.7 CSV proxy label 不能覆盖模型定义

上游 proxy label 是数据或诊断，不是最终六状态定义。

### 20.8 单一特征不能代替多层证据

六状态引擎是 latent state inference，不是单因子规则。

---

## 21. 总结：Feature Contract 是模型可信度的基础

Feature Contract 的价值在于保证五件事：

1. **输入稳定性**：每次运行看到相同类型的证据；
2. **因果正确性**：当前状态只使用当前可得信息；
3. **金融语义一致性**：状态不是由缺失或 fallback 意外改变；
4. **工程可审计性**：缺失、错位、泄漏会显式暴露；
5. **策略接入安全性**：下游策略可以稳定理解输出字段。

一句话总结：**Feature Contract 不是输入 schema，而是六状态清算—吸收模型的因果信息边界。**
