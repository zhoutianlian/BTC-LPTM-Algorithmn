# Output Contract：六状态清算—吸收市场状态引擎的输出说明

> 版本：V2.9 Reader-friendly Output Contract  
> 适用范围：`state_timeseries.csv`、`segment_diagnostics.csv`、Dashboard 以及策略接入所需的关键输出字段。  
> 核心原则：本模型输出的是市场机制底盘，不是单一买卖信号。

---

## 1. 这份文档解决什么问题

**Output Contract（输出契约）** 是本项目对模型结果的正式解释。它说明模型会输出哪些文件、每个字段代表什么金融语义、字段之间如何组合、策略系统如何读取，以及哪些误用必须避免。

这很重要，因为本项目输出不是一个普通分类结果。它是一个多层状态推理系统的最终产物：先有六状态 belief vector，再经过 production stability 得到 `stable_state`，然后进一步输出方向解释、交易触发质量、状态内价格合规、退出风险和段级审计结果。

因此，输出契约的目的不是简单列字段，而是帮助读者理解：

```text
哪些字段解释市场结构；
哪些字段解释交易方向；
哪些字段判断交易触发质量；
哪些字段提示状态失效与风控；
哪些字段用于回测审查和人工复盘。
```

最重要的一点：**模型输出不是直接交易信号**。`stable_state` 不等于买卖指令，`trade_direction` 不等于必须交易，`price_expectation_exit_flag` 也不是机械平仓命令。策略系统应组合多个字段进行判断。

---

## 2. 模型输出总览

当前主要输出包括：

| 输出 | 用途 | 适合使用者 | 主要用途 |
|---|---|---|---|
| `state_timeseries.csv` | 逐时点状态输出 | 策略系统、研究员、风控 | live/回测中读取状态、belief、方向、质量、退出信号 |
| `segment_diagnostics.csv` | 连续状态段审计 | 研究员、模型审查人员 | 检查状态段收益、波动、趋势、质量问题 |
| Dashboard / `six_state_dashboard.html` | 可视化人工审查 | 研究员、交易研究、工程协作者 | 查看价格、状态、belief、PLIE/path/evidence、质量与退出 |
| `segment_quality_flags.csv` / `low_quality_segment_examples.csv` | 低质量段样本 | 模型优化与审计 | 定位弱 HPEM、低质量 VT、RC price drift、ST flat 等问题 |
| `validation_report.*` | 运行校验报告 | 工程与研究 | no-leakage、分布、典型场景、数据质量摘要 |

输出层级可理解为：

```text
Raw filtered belief
→ Production stable state
→ Directional interpretation
→ Trading trigger quality
→ Price expectation compliance
→ Exit / invalidation signals
→ Segment diagnostics
→ Dashboard review
→ Strategy-facing integration
```

---

## 3. `state_timeseries.csv`：逐时点状态输出

### 文件用途

`state_timeseries.csv` 是策略系统最常读取的主输出文件。每一行对应一个 source-clock 推理时点，包含价格、六状态 belief、candidate/stable state、方向解释、交易质量、price expectation、evidence 和质量标记。

### 粒度

粒度通常与 `base_context.csv` 的 source-clock 一致，而不是任意价格 K 线。若上游 PLIE/HMM 约小时级更新，`state_timeseries.csv` 的状态语义也应按 source-clock 理解。

### 适用场景

- 策略条件输入；
- 风控上下文；
- 状态切换审查；
- 指标分桶与回测；
- Dashboard 可视化。

### 不适用场景

- 不应直接把某个字段当成买卖信号；
- 不应把 belief 当收益概率；
- 不应用 future response 或 segment 未来收益解释当前行。

---

## 4. Belief Vector：六状态置信分布

市场状态通常不是瞬间从一种机制跳到另一种机制。一个时点可能同时具有趋势、吸收、冲突和波动证据。因此模型首先输出六个状态上的置信分布，也就是 **belief vector**。

公式上：

```text
b_t = (b_ST, b_VT, b_RC, b_RHA, b_HPEM, b_AMB)
```

直觉上，它表示当前截至 `t` 的全部证据分别支持六种市场机制的程度。六个分量应非负且和为 1。

### `b_ST ... b_AMB`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 六状态 filtered belief 分量。  
**金融解释：** 表示当前清算压力、path、market response、price context 与 quality evidence 对每个状态的支持程度。  
**常见取值：** 0 到 1，六列和约为 1。  
**策略使用方式：** 用于判断状态置信度、构造状态加权策略、观察过渡期。  
**常见误用：** 把 `b_HPEM` 当成上涨/下跌概率；它不是收益概率。  
**与其他字段的关系：** `candidate_state` 通常来自最大 belief；`stable_state` 还要经过稳定层。

### `belief_gap`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 第一高 belief 与第二高 belief 的差距。  
**金融解释：** belief gap 越高，当前主状态与竞争状态的边界越清晰；gap 低说明状态竞争激烈，可能处于过渡或冲突。  
**策略使用方式：** 可作为交易质量过滤条件。低 gap 时降低仓位或 no-trade。  
**常见误用：** 只看 `stable_state`，忽略低 `belief_gap` 下状态语义可能不清。  
**与其他字段的关系：** 与 `candidate_state`、`stable_state` 不一致时尤其重要。

---

## 5. Candidate State 与 Stable State

### `candidate_state`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** categorical  
**含义：** 当前 raw / filtered belief 下最强的候选状态。  
**金融解释：** 更敏感，能更快反映短期证据变化。  
**策略使用方式：** 用于观察潜在状态切换压力，不建议单独作为交易状态。  
**常见误用：** 把 candidate 当最终稳定状态，会重新引入频繁跳变。  
**与其他字段的关系：** 若 `candidate_state != stable_state`，可能表示状态切换正在形成。

### `stable_state`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** categorical  
**含义：** 经 production stability layer 后得到的稳定市场结构状态。  
**金融解释：** 表示当前市场主导机制，例如 HPEM 是清算压力穿透，RHA 是高吸收/拒绝，VT 是主动趋势。  
**策略使用方式：** 用于策略上下文、风控 regime、信号过滤。  
**常见误用：** 直接写 `if stable_state == HPEM: trade()`。stable_state 不是买卖信号。  
**与其他字段的关系：** 应与 `belief_gap`、`directional_state`、`trading_trigger_quality`、`price_expectation_exit_flag` 联合使用。

### `state_duration` / `stable_state_duration`

**所在文件：** `state_timeseries.csv`  
**字段类型：** numeric  
**含义：** 当前 stable_state 已持续的 bar 数或 source-clock steps。  
**金融解释：** 状态持续时间影响切换意义。刚进入的状态可能仍在确认期；持续过久的状态若出现反向证据，应关注退出压力。  
**策略使用方式：** 过滤过短状态、识别 late-stage HPEM/RHA、配合 exit flag。  
**常见误用：** 认为持续越久越安全；有些状态持续久反而可能接近衰竭。

### `transition_reason`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** text  
**含义：** 当前切换或保持状态的主要解释。  
**金融解释：** 帮助追踪状态变化是由 cascade、rejection、active dominance、quality conflict、exit pressure 还是 price expectation 触发。  
**策略使用方式：** 用于人工审查、异常定位、策略调参。  
**常见误用：** 把文本 reason 当作规则引擎输出；它是解释信息，不是交易指令。

### `top_evidence`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** text/list-like  
**含义：** 当前状态判断最重要的 evidence features。  
**金融解释：** 展示状态背后的证据来源，例如 PLIE pressure、path cascade、rejection、active dominance、quality bad。  
**策略使用方式：** 用于解释状态、构造风控审查、识别单一证据过度主导。  
**常见误用：** 只看 top evidence 中的一个字段交易。

---

## 6. 六个状态的输出语义

### ST — Stable Trend

**直觉：** 温和、有序、低冲突的趋势。  
**机制：** 压力或主动趋势存在，但不是极端级联，也不是高吸收拒绝。  
**策略语境：** 可作为有序趋势背景，但仍需方向质量与 price expectation。  
**常见误用：** 把 ST 当极端趋势；ST 应是温和趋势，不是 HPEM。

### VT — Volatile Trend

**直觉：** 主动交易或外生冲击主导的高波趋势。  
**机制：** PLIE 不是主解释变量，价格由 active dominance 或宏观/情绪/现货流推动。  
**策略语境：** 需要进一步看 `VT_UP / VT_DOWN / VT_MIXED`。  
**常见误用：** 把所有 VT 当固定方向；VT 可短期换方向。

### RC — Range-bound / Consolidation

**直觉：** 低压力、低波动、低趋势、区间压缩或等待状态。  
**机制：** 清算压力不强，主动位移不强，市场解释力较低但不一定冲突。  
**策略语境：** 可作为低交易价值或压缩语境；不等于天然可做区间。  
**常见误用：** 只要 RC 就做区间交易；若 `state_price_expectation_issue = RC_PRICE_TOO_LARGE_OR_TRENDING`，RC 假设正在失效。

### RHA — Reversal / High Absorption

**直觉：** 清算压力仍在，但价格不再服从。  
**机制：** forced flow 被吸收、拒绝，或反向接管。  
**策略语境：** 要区分 `RHA_REVERSAL_UP/DOWN`、`RHA_STALL`、`RHA_WEAK`。  
**常见误用：** 把所有 RHA 当反转交易。RHA_STALL 可能只是吸收，不具备明确方向。

### HPEM — High Pressure / Extreme Move

**直觉：** 清算压力穿透市场，价格顺压力方向级联或极端位移。  
**机制：** directional PLIE + cascade + price impact / jump / high vol。  
**策略语境：** 需要看 `HPEM_UP / HPEM_DOWN`、directional quality、giveback、price expectation。  
**常见误用：** 把 HPEM 当高波动。高波动但 PLIE 不主导时更可能是 VT。

### AMB — Ambiguous / No-trade

**直觉：** 证据冲突、低质量或不适合交易的保护性状态。  
**机制：** path/memory/PLIE/price context 冲突，或数据质量风险。  
**策略语境：** 通常作为 no-trade / 降权 / 风控状态。  
**常见误用：** 把 AMB 当模型失败。AMB 是保护层。

---

## 7. Directional Output：方向解释层

`stable_state` 回答“市场机制是什么”，但交易还需要知道这个机制是否具有可解释方向。因此模型输出 directional layer。

### `directional_state`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** categorical  
**含义：** 带状态语义的方向解释。  
**金融解释：** 例如 `HPEM_DOWN` 表示 HPEM 的方向为向下，且由 PLIE/价格合规支持；`VT_MIXED` 表示 VT 内部方向混合。  
**常见取值：** `HPEM_UP`, `HPEM_DOWN`, `WEAK_HPEM`, `RHA_REVERSAL_UP`, `RHA_REVERSAL_DOWN`, `RHA_STALL`, `RHA_WEAK`, `INVALID_RHA_FOLLOWS_PLIE`, `VT_UP`, `VT_DOWN`, `VT_MIXED`, `RC_NO_TRADE`, `ST_UP`, `ST_DOWN`, `AMB` 等。以项目实际输出枚举为准。  
**策略使用方式：** 用于判断状态是否存在方向交易解释。  
**常见误用：** 把 directional_state 当无条件开仓信号。

### `trade_direction`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** categorical  
**含义：** 简化方向枚举。  
**常见取值：** `UP`, `DOWN`, `MIXED`, `NO_TRADE`。  
**金融解释：** 只表达方向，不表达状态机制。  
**策略使用方式：** 策略可用它决定方向候选，但必须配合 quality 与 exit flag。  
**常见误用：** 看到 `UP` 就做多；它还需要状态质量与风险过滤。

### `direction_confidence`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 方向证据强弱。  
**金融解释：** 越高表示趋势方向、PLIE alignment、path active direction 或 reversal evidence 更清楚。  
**策略使用方式：** 方向过滤；低值时不做方向交易或降低权重。  
**常见误用：** 当成收益概率。

### `directional_trading_quality`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 方向交易解释质量。  
**金融解释：** 衡量当前方向状态是否具备足够一致性、价格合规和证据支持。  
**策略使用方式：** 过滤 `WEAK_HPEM`、`RHA_STALL`、`VT_MIXED` 等低质量方向状态。  
**常见误用：** 当作收益承诺。

### `directional_no_trade_reason`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** text  
**含义：** 当前方向层不适合交易的原因。  
**金融解释：** 例如方向混合、HPEM giveback、RHA 只是 stall、价格预期失效。  
**策略使用方式：** 用于风控解释与人工审查。

---

## 8. Trading Trigger Quality：状态是否适合交易触发

状态成立不等于交易触发成立。模型可能识别到 RHA 语境，但价格尚未形成反向接管；也可能识别到 HPEM 压力，但价格穿透不足。因此需要交易触发质量层。

### `state_confirmation_score`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 当前状态进入以来，已发生价格行为与该状态定义的匹配程度。  
**金融解释：** HPEM 需要穿透，RC 需要低波压缩，VT 需要有效位移，ST 需要温和趋势。  
**策略使用方式：** 作为状态确认过滤。  
**常见误用：** 认为高 score 等于收益更高；它只表示状态语义更合规。

### `state_confirmation_status`

**所在文件：** `state_timeseries.csv`  
**字段类型：** categorical  
**含义：** 状态确认状态。  
**常见取值：** `confirmed`, `pending`, `weak`, `unconfirmed` 等，以实际输出为准。  
**金融解释：** pending/weak 表示市场结构可能存在，但交易解释还不充分。

### `trading_trigger_quality`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 当前 stable_state 是否适合作为交易触发上下文。  
**金融解释：** 结合状态确认、方向质量、价格行为和证据质量。  
**策略使用方式：** 推荐作为交易信号过滤的核心字段之一。  
**常见误用：** 把它当成胜率或收益预期。

### `weak_state_flag`

**所在文件：** `state_timeseries.csv`  
**字段类型：** boolean / 0-1  
**含义：** 当前状态语义存在，但交易质量不足。  
**金融解释：** 例如 weak HPEM：清算压力存在，但价格穿透不足。  
**策略使用方式：** 降权、观察、风控，不建议作为主动开仓触发。

---

## 9. Price Expectation Compliance：状态价格行为是否仍然合规

状态进入后，价格行为必须继续符合状态定义。V2.5 引入 price expectation compliance，用于检查状态内价格生命周期。

### `state_price_expectation_score`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 当前状态段内价格行为是否符合该状态的价格预期。  
**金融解释：** HPEM 应顺 PLIE 方向穿透；RC 应价格变化小；VT 应有较大位移；ST 应温和有序；RHA 应表现为反 PLIE 接管或吸收。  
**策略使用方式：** 低分时降低交易权重或观察状态失效。  
**常见误用：** 当成未来收益评分；它只评价已发生的状态内行为。

### `state_price_expectation_issue`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** categorical/text  
**含义：** 价格预期不合规的具体类型。  
**常见取值：** `OK`, `HPEM_PRICE_EXPECTATION_FAIL`, `RHA_INVALID_FOLLOWS_PLIE`, `RHA_WEAK_NO_REVERSAL`, `RC_PRICE_TOO_LARGE_OR_TRENDING`, `VT_LOW_MOVE`, `VT_MIXED_DIRECTION`, `ST_FLAT`, `ST_WEAK_TREND`, `ST_TOO_EXTREME` 等，以实际输出为准。  
**策略使用方式：** 用于 no-trade、减仓、退出、人工审查。

### `price_expectation_exit_flag`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** boolean / 0-1  
**含义：** 当前 stable_state 的价格行为出现失效或风险上升。  
**金融解释：** 它提示当前状态的交易含义可能不再成立，例如 HPEM giveback、RHA 重新服从 PLIE、RC 出现趋势漂移。  
**策略使用方式：** 用于减仓、退出、降低权重、观察下一状态。  
**常见误用：** 当作机械平仓指令。它是风险提示，不是单一执行命令。  
**与其他字段的关系：** 应与 `price_expectation_exit_target`、`exit_reason`、belief 和 directional quality 一起看。

### `price_expectation_exit_target`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** categorical/text  
**含义：** 当前状态失效后更可能转向的候选状态。  
**金融解释：** 例如 HPEM decay 后若 rejection/absorption 高，exit target 可能为 RHA；RC price drift 后若 active/trend 高，target 可能为 VT。  
**策略使用方式：** 作为观察下一状态或调整风险的参考。  
**常见误用：** 当成下一状态必然发生。

### `price_expectation_exit_reason`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** text  
**含义：** exit flag 的具体原因。  
**策略使用方式：** 用于人工复盘和风控解释。

### `price_expectation_trading_quality`

**所在文件：** `state_timeseries.csv` / Dashboard  
**字段类型：** numeric  
**含义：** 价格预期层面对交易质量的评估。  
**金融解释：** 即使 stable_state 成立，若价格行为不再合规，该质量应下降。  
**策略使用方式：** 与 `trading_trigger_quality` 共同过滤。

---

## 10. Exit / Invalidation Signals：状态失效与风控输出

exit / invalidation signals 的目的不是替代稳定层，而是告诉策略：当前稳定状态的金融假设是否正在被价格行为或证据结构证伪。

常见使用方式：

```text
price_expectation_exit_flag == 1:
    观察当前状态失效；
    降低原方向交易权重；
    根据 exit_target 审查下一状态；
    不机械执行，需结合策略规则。
```

示例：

```text
stable_state = HPEM
price_expectation_exit_flag = 1
price_expectation_exit_reason = HPEM aligned MFE gave back beyond threshold
```

这表示清算压力穿透的交易含义正在衰减，策略可考虑减仓或等待 RHA/VT/AMB 确认，但它不是自动平仓命令。

---

## 11. `segment_diagnostics.csv`：状态段级审计输出

### 文件用途

`segment_diagnostics.csv` 将逐点 `stable_state` 压缩成连续状态段，用于回测后审计、状态质量评估、状态定义校验、人工复盘和 walk-forward 稳定性检查。

### 关键字段说明

#### `segment_id`

**字段类型：** numeric/id  
**含义：** 连续状态段编号。  
**策略使用方式：** 用于把逐点状态聚合到段级审查。

#### `state`

**字段类型：** categorical  
**含义：** 该段的 stable_state。  
**金融解释：** 代表该连续区间的主导市场机制。

#### `start_time` / `end_time`

**字段类型：** timestamp  
**含义：** 状态段开始与结束时间。  
**策略使用方式：** 用于复盘和与价格图对齐。

#### `duration_bar_count`

**字段类型：** numeric  
**含义：** 状态段持续 bar 数。  
**金融解释：** 过短段可能是噪声或 provisional 状态；过长段若 price expectation 失效则需要审查。

#### `entry_price` / `exit_price`

**字段类型：** numeric  
**含义：** 状态段进入和退出价格。  
**注意：** 这是段级审计字段，不是实时当前行的未来信息。不能在进入时点使用未来 exit_price。

#### `segment_return_bps`

**字段类型：** numeric  
**含义：** 状态段从进入到退出的净价格变化，单位 bps。  
**金融解释：** 用于判断该状态段价格行为是否符合状态定义。  
**常见误用：** 把它当策略收益。它不是策略收益，因为没有考虑入场/出场规则、方向、成本、滑点和仓位。

#### `segment_range_bps`

**字段类型：** numeric  
**含义：** 状态段内最高价与最低价的区间宽度，单位 bps。  
**金融解释：** 用于判断 RC 是否真小幅、VT/HPEM 是否有足够位移。

#### `max_favorable_move_bps`

**字段类型：** numeric  
**含义：** 按该段方向解释计算的最大有利移动。  
**金融解释：** 衡量状态段内潜在方向移动，但不是可实现收益。  
**常见误用：** 当作策略最大收益。

#### `max_adverse_move_bps`

**字段类型：** numeric  
**含义：** 按该段方向解释计算的最大不利移动。  
**金融解释：** 用于评估状态段内风险暴露。

#### `mean_realized_vol_1h_bps`

**字段类型：** numeric  
**含义：** 段内 1h realized vol 均值。  
**金融解释：** 审查 HPEM/VT 是否具有足够波动，RC/ST 是否过度波动。

#### `mean_trend_strength_6h`

**字段类型：** numeric  
**含义：** 段内 6h trend strength 均值。  
**金融解释：** 审查 VT/ST 是否趋势充分，RC 是否出现不该有的趋势。

#### `dominant_path_context_6h` / `dominant_path_label_6h`

**字段类型：** categorical  
**含义：** 段内占主导的 6h path context / label。  
**金融解释：** 帮助判断该段是 directional pressure、neutral active、quiet 还是 rejection/cascade。

#### `segment_quality_issues`

**字段类型：** text/list-like  
**含义：** 该状态段发现的质量问题。  
**金融解释：** 例如 HPEM no-response、RHA invalid follows PLIE、RC price too large、VT low move、ST flat。  
**策略使用方式：** 用于模型审计、参数优化、回测分桶，不用于单点 live 触发。

---

## 12. Dashboard：人工审查与解释工具

Dashboard 用于人工审查，不参与模型判断。它适合展示：

- 价格与 stable_state 背景；
- belief vector；
- PLIE / HMM / path / price context；
- state confirmation；
- HPEM giveback；
- RHA subtype；
- VT direction；
- price expectation exit flags；
- segment quality table。

Dashboard 的价值在于解释和审计，而不是反过来改变模型输出。它适合研究复盘、异常检查、状态切换解释和策略接入调试。

---

## 13. 下游策略如何组合使用输出

策略不应只读取 `stable_state`。推荐至少组合：

```text
stable_state                  市场结构
belief_gap                    状态置信边际
directional_state             方向解释
trade_direction               简化方向
directional_trading_quality   方向质量
trading_trigger_quality       交易触发质量
state_price_expectation_score 状态内价格合规
price_expectation_exit_flag   状态失效/退出风险
```

错误示例：

```python
if stable_state == "HPEM":
    long_or_short()
```

正确的组合逻辑示例：

```python
if (
    stable_state == "HPEM"
    and directional_state in {"HPEM_UP", "HPEM_DOWN"}
    and belief_gap >= threshold
    and directional_trading_quality >= threshold
    and state_price_expectation_score >= threshold
    and price_expectation_exit_flag == 0
):
    # consider trend-following / risk-expansion context
    pass
```

这只是字段组合逻辑示例，不是收益承诺，也不是具体投资建议。

---

## 14. 输出字段之间的关系

### belief vector vs stable_state

belief vector 是连续置信分布；stable_state 是经过稳定层之后的离散状态。

### candidate_state vs stable_state

candidate_state 更敏感；stable_state 更稳健。二者不一致可能表示状态切换正在形成。

### stable_state vs directional_state

stable_state 解释市场结构；directional_state 解释交易方向质量。

### directional_state vs trade_direction

directional_state 是带状态语义的方向解释；trade_direction 是简化方向枚举。

### trading_trigger_quality vs price_expectation_trading_quality

trading_trigger_quality 关注当前状态是否适合交易触发；price_expectation_trading_quality 关注状态段内价格行为是否继续符合该状态定义。

### price_expectation_exit_flag vs 状态切换

exit flag 不一定立刻改变 stable_state。它表示当前状态价格行为出现失效或风险上升，策略可用于减仓、退出、观察或降低信号权重。

---

## 15. 常见误用与风险

### 15.1 把 stable_state 当作直接买卖信号

错误。stable_state 是市场机制状态，不是买卖指令。应结合方向、质量、belief gap 和 exit flag。

### 15.2 把 HPEM 当作无条件追涨杀跌

错误。HPEM 必须看方向、价格穿透、giveback、price expectation。`WEAK_HPEM` 或 exit flag 表示不能无条件追随。

### 15.3 把所有 RHA 当作反转交易

错误。RHA 包含 reversal、stall、weak、invalid follow PLIE。只有 reversal 子型更接近方向交易语境。

### 15.4 忽略 belief_gap

低 belief gap 表示状态竞争激烈，策略应更谨慎。

### 15.5 忽略 directional_state

尤其对 VT，必须区分 VT_UP、VT_DOWN、VT_MIXED。

### 15.6 忽略 trading_trigger_quality

状态识别成立，不代表交易触发质量足够。

### 15.7 忽略 price_expectation_exit_flag

当状态内价格行为失效时，继续沿原状态交易可能放大风险。

### 15.8 把 segment_return_bps 当作策略收益

segment_return_bps 是状态段审计指标，不包含策略入场、方向、成本、滑点和仓位。

### 15.9 把 Dashboard 当成模型输入

Dashboard 是解释工具，不参与模型推理。

### 15.10 把 AMB 理解为模型失败

AMB 是保护性 no-trade 状态，表示证据冲突或质量不足。

### 15.11 只看单点输出，不看连续状态段

状态质量需要结合持续时间、segment diagnostics 和 price expectation 审查。

### 15.12 在回测中使用未来字段解释输出

不能用未来 response、未来收益或 segment 终点信息解释当前 live 时点。

---

## 16. 输出质量与审计建议

建议每次运行后审查：

```text
belief gap / belief entropy
state duration distribution
transition reason audit
state distribution stability
segment_quality_issues
HPEM / RHA / VT / RC / ST 的 price expectation issue
OOS / walk-forward 稳定性
与 Feature Contract 的输入一致性
```

如果状态分布突变，应先检查输入契约和 no-leakage 报告，再判断是否为市场 regime 变化。

---

## 17. 总结：如何正确理解本项目输出

本项目输出应被理解为：

```text
市场机制描述
+ 策略上下文
+ 方向解释
+ 交易质量过滤
+ 状态失效风控
+ 审计工具
```

而不是：

```text
单一买卖信号
收益预测概率
机械开平仓指令
```

最重要的使用原则是：

```text
stable_state                  看市场结构；
directional_state             看交易方向；
trading_trigger_quality       看是否值得触发；
state_price_expectation_score 看状态内价格行为是否合规；
price_expectation_exit_flag   看状态是否正在失效；
belief_gap                    看置信边际。
```

一句话总结：**Output Contract 不是字段清单，而是策略系统正确理解六状态引擎输出的语义边界。**
