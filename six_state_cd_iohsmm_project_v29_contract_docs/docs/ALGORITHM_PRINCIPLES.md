# 六状态清算—吸收市场状态引擎：读者友好的算法原理说明

**项目：** QuantAlgorithm — BTC 清算压力、吸收率与市场状态推理  
**文档定位：** 面向量化研究员、策略研究人员、机器学习工程师和项目协作者的算法原理说明  
**当前版本：** V2.7 reader-friendly algorithm document，基于 V2.6 strict input contract 与 V2.5 price expectation compliance  

---

## 0. 一页摘要：这套模型应该怎样被理解

这套模型不是用来预测下一根 K 线涨跌的，也不是把市场粗暴分成“上涨、下跌、震荡”的分类器。它要解决的问题更接近：

> 当前 BTC 杠杆市场中，推动价格的主导机制是什么？  
> 是清算压力正在穿透市场，还是这股压力正在被吸收？  
> 是主动资金在主导趋势，还是证据冲突、暂时不适合交易？

模型围绕一条核心链路建模：

```text
清算压力 → 价格响应 → 吸收 / 传导 / 放大 / 拒绝 → 市场状态
```

在 BTC 衍生品市场里，清算不是普通交易。普通交易来自交易者的主动判断；清算来自保证金约束，是交易者“不得不做”的强制交易。清算量告诉我们哪一侧出现了 forced flow；价格响应告诉我们市场有没有接住、放大、拒绝或反向接管这股强制流。

因此，本模型输出的不是买卖指令，而是市场机制底盘。它在每个时点输出六个状态上的置信分布，也就是 **belief vector**：

$$
b_t = \left(
 b_t^{ST}, b_t^{VT}, b_t^{RC}, b_t^{RHA}, b_t^{HPEM}, b_t^{AMB}
\right),
\qquad b_t^k \ge 0,
\qquad \sum_k b_t^k = 1.
$$

这里的六个状态分别是：

| 状态 | 全称 | 直觉含义 |
|---|---|---|
| `ST` | Stable Trend | 温和、有序、低冲突的趋势。 |
| `VT` | Volatile Trend | 主动交易或外生冲击主导的高波动趋势，清算压力不是主解释变量。 |
| `RC` | Range-bound / Consolidation | 低压力、低波动、低趋势、区间压缩或等待环境。 |
| `RHA` | Reversal / High Absorption | 清算压力仍在，但价格不再服从，出现吸收、拒绝或反向接管。 |
| `HPEM` | High Pressure / Extreme Move | 清算压力穿透市场，forced flow 与价格路径共振，可能形成 squeeze / flush / cascade。 |
| `AMB` | Ambiguous / No-trade | 证据冲突、数据质量不足或不适合高置信交易判断的保护状态。 |

实盘策略不应只读取 `stable_state`。更合理的策略输入至少包括：

```text
stable_state                  # 市场结构语境
directional_state             # 交易方向解释
trade_direction                # UP / DOWN / MIXED / NO_TRADE
trading_trigger_quality        # 当前状态是否适合交易触发
state_price_expectation_score  # 状态内价格行为是否仍符合定义
price_expectation_exit_flag    # 当前状态是否出现价格行为失效
belief_gap                     # 最强状态与次强状态的置信边际
```

一句话总结：

> `stable_state` 用来理解市场机制；`directional_state` 用来理解方向；`trading_trigger_quality` 用来判断是否值得交易；`price_expectation_exit_flag` 用来管理状态失效和退出风险。

---

## 1. 这套模型到底在解决什么问题

### 1.1 它不是在预测下一根 K 线

如果把问题定义成“预测下一根 K 线上涨还是下跌”，模型会被迫关注短期噪声。BTC 市场 24/7 交易、杠杆高、交易所结构复杂，短期价格变化常常同时受到宏观消息、现货流、期货杠杆、资金费率、强平链条和流动性断层影响。单纯预测未来收益，很容易把偶然价格路径当成可交易规律。

本模型换了一个问题：

```text
不是问：下一根 K 线涨还是跌？
而是问：当前价格运动背后的主导机制是什么？
```

例如，同样是下跌 200 bps：

- 如果多头清算强烈、价格继续顺着 forced selling 下跌，这是 HPEM 的候选场景；
- 如果多头清算仍然明显，但价格跌不动甚至反弹，这是 RHA 的候选场景；
- 如果清算压力中性，价格因为主动卖盘或外生冲击下跌，这是 VT 的候选场景；
- 如果证据冲突或数据质量不可靠，应进入 AMB，而不是强行交易。

这就是状态模型与收益预测模型的根本差别。

### 1.2 它也不是简单市场分类器

普通分类器通常学习一个函数：

$$
z_t = f(x_t).
$$

这表示状态只由当前一行特征决定。但市场状态通常不是瞬间从一种机制跳到另一种机制。HPEM 可能先放大，再进入 RHA；RHA 可能先是吸收停滞，随后才反转；VT 可能在同一个高波动阶段里短期方向切换。若只看当前一行特征，模型会容易频繁跳变。

本项目的状态推理目标是：

$$
P(z_t \mid x_{1:t}),
$$

也就是状态依赖从过去到现在的整条证据路径。模型要同时考虑：

```text
上一状态是什么；
这个状态已经持续多久；
当前 PLIE / HMM 给出的清算压力是什么；
最近成熟 market response 如何裁决清算事件；
6h / 12h / 24h / 48h path absorption 如何描述中期路径；
当前价格波动、趋势和跳跃是否支持状态切换；
证据是否冲突、数据是否可靠。
```

### 1.3 它要识别的是市场主导机制

本模型输出六状态，不是为了给 K 线形态命名，而是为了识别市场内部力量关系：

```text
清算压力是否存在？
压力是否穿透了价格？
forced flow 是否被吸收？
价格运动是否由主动交易主导？
市场是否进入高压力极端运动？
证据是否混乱到不适合交易？
```

因此，模型的最终价值不是“预测一个方向”，而是为交易策略提供更高层的语境：什么状态下可以顺势，什么状态下应等待，什么状态下应减仓，什么状态下应避免把噪声当成信号。

---

## 2. 核心直觉：清算压力只是起点，价格响应才是裁决

### 2.1 为什么 BTC 杠杆市场要看清算

BTC 衍生品市场里，大量仓位使用杠杆。杠杆的本质不是放大收益，而是压缩容错空间。当价格朝不利方向移动时，高杠杆仓位会更快接近维持保证金阈值。一旦账户权益无法满足维持保证金要求，清算引擎会强制平仓。

清算的关键特点是：

```text
它不是交易者主动表达观点；
它是风险约束触发后的机械执行；
它通常没有等待权、议价权和择时权；
它会把潜在脆弱性转化为真实成交压力。
```

这就是为什么清算量比很多情绪变量更接近结果层事实。它记录的是“谁已经被市场强制推出去”。

### 2.2 forced buying 与 forced selling

在永续或期货市场里，多头清算和空头清算的价格压力方向相反：

```text
空头清算 = 空头被迫买回 = forced buying = 向上压力
多头清算 = 多头被迫卖出 = forced selling = 向下压力
```

这个方向约定非常重要。不能把“空头清算”误解为价格向下，也不能把“多头清算”误解为价格向上。

对窗口 $[t,t+\Delta]$，记空头清算规模为 $L^{short}_{t,\Delta}$，多头清算规模为 $L^{long}_{t,\Delta}$。净清算流定义为：

$$
Q^{liq}_{t,\Delta}=L^{short}_{t,\Delta}-L^{long}_{t,\Delta}.
$$

这个公式的直觉很简单：空头清算带来买入压力，多头清算带来卖出压力，两者相减后得到净 forced flow。

当 $Q^{liq}_{t,\Delta}>0$ 时，净 forced buying 占优；当 $Q^{liq}_{t,\Delta}<0$ 时，净 forced selling 占优。

进一步定义清算主导度：

$$
\Phi^{liq}_{t,\Delta}
=
\frac{L^{short}_{t,\Delta}-L^{long}_{t,\Delta}}
{L^{short}_{t,\Delta}+L^{long}_{t,\Delta}+\varepsilon},
\qquad \varepsilon>0.
$$

它表示在总清算活动中，哪一侧更占主导。相比净流量，主导度更接近方向性压力结构。

再定义总清算强度：

$$
I^{liq}_{t,\Delta}
=
\log\left(1+L^{short}_{t,\Delta}+L^{long}_{t,\Delta}\right).
$$

它表示这段时间去杠杆活动有多剧烈。这里取对数，是为了减弱极端清算规模的数值支配效应，让模型更稳健。

### 2.3 为什么单看清算量会误判

清算量只告诉我们压力出现了，但不告诉我们压力是否真正穿透市场。

举例：同样是大规模多头清算，也就是 forced selling：

- 如果价格继续快速下跌，说明市场没有接住这批被迫卖盘，压力被传导甚至放大；
- 如果价格几乎不跌，说明市场中存在足够承接；
- 如果价格不跌反升，说明旧方向压力可能被吸收，甚至被反向接管。

所以，清算量是压力源，价格响应才是市场裁决。

记响应窗口 $[t,t+h]$ 上的价格收益为：

$$
r_{t,h}=\log\left(\frac{P_{t+h}}{P_t}\right).
$$

为了判断价格是否服从清算方向，可以定义方向性响应：

$$
S_{t,h}=\operatorname{sign}\left(Q^{liq}_{t,\Delta}\right)\cdot r_{t,h}.
$$

这个公式的直觉是：如果价格沿清算压力方向移动，$S_{t,h}>0$；如果价格逆清算压力方向移动，$S_{t,h}<0$。它回答的问题不是“价格涨跌”，而是“价格是否服从这股 forced flow”。

### 2.4 吸收、传导、级联与拒绝

模型最关心的是清算发生后价格如何响应。可以把响应分成几类：

```text
传导 transmission：价格沿清算压力方向移动。
级联 cascade：传导很强，价格运动继续触发更多清算。
吸收 absorption：清算压力出现，但价格位移弱于预期。
拒绝 rejection：清算压力仍在，但价格不再服从。
反向接管 takeover：价格明显逆清算压力方向运动，主导权切换。
```

这几类响应不是文学描述，而是状态区分的核心。HPEM、RHA、ST 的差别，很多时候不在于清算压力有没有出现，而在于价格对清算压力的处理方式。

---

## 3. 六个状态的直觉解释

本节用更接近交易和市场结构的语言解释六个状态。需要特别强调：这些状态不是涨跌标签，也不是交易指令。

### 3.1 ST — Stable Trend / 稳定趋势

**直觉解释：**  
ST 是温和、有序、低冲突的趋势状态。市场存在方向，但不是极端清算级联，也不是被高吸收拒绝。价格大体服从当前结构，波动不过度失控。

**清算—价格关系：**  
清算压力可能存在，但传导较有序。常见的是 baseline transmission 或 partial absorption：价格跟随压力或趋势推进，但没有进入 HPEM 那种强穿透和极端放大。

**典型市场表现：**

```text
趋势方向较清晰；
波动中等；
PLIE reliability 较好；
path_label 偏 baseline_transmission / partial_absorption；
证据冲突低。
```

**策略含义：**  
ST 可作为趋势策略的背景状态，但不等于无条件追随。策略仍需检查 `directional_state`、`trend_consistency`、`trading_trigger_quality` 和成本后的收益空间。

**常见误用：**  
把 ST 当成强趋势。ST 不是 HPEM，不能期待极端位移；ST 更像“有序推进”，而不是“爆发行情”。

---

### 3.2 VT — Volatile Trend / 波动趋势

**直觉解释：**  
VT 是主动交易或外生冲击主导的高波动趋势。它的关键不是清算压力穿透，而是价格自己在强烈移动。

**清算—价格关系：**  
PLIE 通常中性、低可靠或不是主解释变量。价格可能由现货流、宏观事件、ETF 资金、主动交易、情绪或其他外生冲击推动。

**典型市场表现：**

```text
path_context 偏 neutral_pressure 或 mixed_pressure；
path_label 偏 active_dominance_up/down 或 mixed_active_breakout；
realized_vol、jump_proxy、trend_strength 较高；
PLIE 解释力低或方向不主导。
```

**策略含义：**  
VT 需要方向层。模型会输出 `VT_UP`、`VT_DOWN` 或 `VT_MIXED`。如果是 `VT_MIXED`，说明市场处于高波动主动环境，但方向不稳定，不应被直接当成方向交易信号。

**常见误用：**  
把 VT 当成 HPEM。两者都可能高波动，但 HPEM 必须有清算压力穿透；VT 则是主动交易或外生冲击主导。

---

### 3.3 RC — Range-bound / Consolidation / 区间震荡或整理

**直觉解释：**  
RC 是低压力、低波动、低趋势、区间压缩或等待状态。市场没有明显 forced-flow 压力，也没有强主动趋势。

**清算—价格关系：**  
清算压力弱或中性，价格路径也没有有效位移。RC 不是“行情消失”，而是当前清算—价格关系缺乏明确交易方向。

**典型市场表现：**

```text
path_context = path_neutral_pressure；
path_label = path_quiet_no_pressure；
realized_vol 低；
range_compression 高；
trend_strength 低；
jump_proxy 低；
conflict 低。
```

**策略含义：**  
RC 可以用于降低趋势策略暴露，也可以作为等待状态。但 RC 不等于一定适合区间交易，因为区间交易还需要额外确认边界、流动性和执行成本。

**常见误用：**  
把所有低波动都当成 RC。低波动也可能是缓慢趋势，所以 RC 还需要低 trend_strength、低 trend_consistency 和较高 range_compression。

---

### 3.4 RHA — Reversal / High Absorption / 反转或高吸收

**直觉解释：**  
RHA 表示清算压力仍然存在，但价格不再服从这股压力。它可以是反转，也可以只是高吸收或停滞。

**清算—价格关系：**  
RHA 必须有明确的方向性清算压力背景。它不是“价格反了就叫 RHA”，而是“清算压力仍在，但价格拒绝它”。

**典型市场表现：**

```text
path_context = path_directional_core / path_directional_weak；
path_label = path_pressure_rejection / path_reversal_takeover / path_full_absorption_stall；
absorption / rejection / takeover evidence 上升；
active_force_price 与 PLIE 方向相反；
价格可能反向，也可能只是横住不再服从。
```

**策略含义：**  
RHA 需要方向质量层。`RHA_REVERSAL_UP/DOWN` 才具备更明确方向交易价值；`RHA_STALL` 只表示吸收，不一定适合方向交易；`RHA_WEAK` 应降低权重；`INVALID_RHA_FOLLOWS_PLIE` 表示 RHA 含义被证伪。

**常见误用：**  
把所有 RHA 都当成反转交易。RHA 的原始状态含义包含 high absorption，不保证立即反转。

---

### 3.5 HPEM — High Pressure / Extreme Move / 高压力极端运动

**直觉解释：**  
HPEM 是清算压力穿透市场，价格与 forced flow 共振的状态。它是最接近 squeeze、flush 或 cascade 的状态。

**清算—价格关系：**  
PLIE 有方向且强，价格顺着清算压力方向走，吸收不足，传导或放大明显。

**典型市场表现：**

```text
path_label = path_cascade_transmission；
PLIE intensity 高；
plie_accel_pos 高；
plie_strong_entry 可能触发；
path_transmission_ratio 高；
path_absorption_score 低；
realized_vol / jump_proxy 高；
价格顺 PLIE 方向形成连续位移。
```

**策略含义：**  
HPEM 不是“高波动”本身，而是清算压力穿透。策略不能只看 `stable_state == HPEM`，还应检查 `directional_state = HPEM_UP/DOWN`、`directional_trading_quality`、`state_price_expectation_score` 和 `price_expectation_exit_flag`。如果出现 `WEAK_HPEM` 或 `HPEM_PRICE_EXPECTATION_FAIL`，不应作为高质量追随信号。

**常见误用：**  
把所有高波动都当成 HPEM。高波动如果来自主动交易而不是 PLIE 穿透，更可能是 VT。

---

### 3.6 AMB — Ambiguous / No-trade / 模糊或观望

**直觉解释：**  
AMB 是保护性状态，不是模型失败。它表示证据冲突、质量不足、压力混合或解释力不够，不适合高置信交易。

**清算—价格关系：**  
可能出现 PLIE、path、market response 和 price context 互相矛盾；也可能是数据质量或时钟质量不足。

**典型市场表现：**

```text
path_context = path_mixed_pressure；
path_label = path_mixed_pressure_chop；
liq_entropy 高；
hmm_conf 低；
mar_response_conflict_score 高；
belief_gap 低；
data_gap_flag 或 staleness_flag 上升。
```

**策略含义：**  
AMB 用于保护策略避免在证据冲突区间过度交易。它可以降低仓位、暂停开仓或触发人工审查。

**常见误用：**  
把 AMB 当成“没有行情”。AMB 不表示市场安静，而表示当前证据不足以支持高置信方向判断。

---

## 4. 为什么模型输出 belief vector，而不是单一标签

### 4.1 市场状态常处在过渡中

市场机制不是开关。HPEM 可能正在衰减为 RHA，RHA 可能正在转向 VT，RC 可能正在被 active breakout 打破。此时输出单一标签会掩盖过渡期的不确定性。

所以模型输出六维 belief vector：

$$
b_t = \left(
 b_t^{ST}, b_t^{VT}, b_t^{RC}, b_t^{RHA}, b_t^{HPEM}, b_t^{AMB}
\right),
\qquad b_t^k \ge 0,
\qquad \sum_k b_t^k = 1.
$$

直觉上，$b_t^k$ 表示截至时点 $t$ 的所有已知证据支持状态 $k$ 的程度。

### 4.2 belief gap 为什么重要

如果最高 belief 是 0.72，第二高是 0.18，模型较有把握。若最高是 0.36，第二高是 0.33，说明状态边界很近，交易上应更谨慎。

定义：

$$
belief\_gap_t = \max_k b_t^k - \max_{j \ne k^*} b_t^j,
\qquad k^*=\arg\max_k b_t^k.
$$

直觉上，belief gap 是状态置信边际。它不是收益概率，而是机制判断的清晰度。

### 4.3 stable_state 为什么不是直接交易信号

`stable_state` 是 production stability layer 后的市场结构标签。它解决的是“当前市场机制更像哪一类”。但交易还需要回答：

```text
这个状态有没有明确方向？
这个方向质量够不够？
当前价格行为是否仍符合该状态定义？
状态是否正在失效？
成本、滑点、仓位风险是否允许交易？
```

所以策略不能只写：

```python
if stable_state == "HPEM":
    trend_follow()
```

更合理的是：

```python
if (
    stable_state == "HPEM"
    and directional_state in {"HPEM_UP", "HPEM_DOWN"}
    and directional_trading_quality >= threshold
    and state_price_expectation_score >= threshold
    and price_expectation_exit_flag == 0
    and belief_gap >= threshold
):
    consider_trend_follow()
```

---

## 5. 模型整体流程

本模型可以理解为一条从“原始特征”到“可交易语境”的多层流水线：

```text
Fixed Input Bundle
→ Feature Contract + No-leakage Canonical Frame
→ Semantic Evidence Layer
→ CD-IOHSMM-lite Filtered Belief Engine
→ Production Stability Layer
→ Responsiveness / Exit Pressure Layer
→ Causal State Confirmation + Directional Quality
→ Price Expectation Compliance Layer
→ State / Direction / Trading Quality Outputs
```

### 5.1 Fixed Input Bundle

为了让每次运行可比较，当前系统使用 strict fixed input bundle。也就是说，输入文件不再是可选项。每次运行都必须提供相同角色的核心 CSV。

必需文件包括：

```text
base_context.csv
plie_predictions_source.csv
absorption_memory.csv
path_absorption_multiscale.csv
path_absorption.csv
price_context_features.csv
liqprice_features.csv
features_liq_dataflow.csv
```

这样做的目的不是增加复杂度，而是避免某次运行因为缺少辅助特征而静默降级，从而导致模型结果不可比较。

### 5.2 Feature Contract + No-leakage Canonical Frame

原始 CSV 来自不同模块，更新频率不同。模型不能直接把它们横向拼接，否则容易出现未来函数或时间错配。Canonical feature frame 的作用是把所有特征对齐到同一个因果时钟：每一行代表时点 $t$ 已经可得的信息。

记 $\mathcal{L}_t$ 为时点 $t$ 可得的清算信息集合，$\mathcal{P}_t$ 为价格信息集合，则观测向量为：

$$
x_t=\mathcal{G}(\mathcal{L}_t,\mathcal{P}_t).
$$

这个公式的直觉是：模型看到的不是原始字段堆叠，而是清算与价格关系的结构化观测。

### 5.3 Semantic Evidence Layer

原始字段很多。模型先把它们压缩成金融语义证据，例如：方向性压力、级联、吸收、拒绝、主动主导、安静整理、冲突和低质量。

这一步的意义是：让后续状态模型不直接依赖杂乱字段，而是依赖可解释的机制证据。

### 5.4 CD-IOHSMM-lite Filtered Belief Engine

CD-IOHSMM-lite 是当前实现中的核心状态推理引擎。它保留了 CD-IOHSMM 的关键思想：状态依赖输入、上一状态、持续时间和转移约束，但以可审计、工程可落地的方式实现。

它输出 raw belief：

$$
b_t=P(z_t \mid x_{1:t}).
$$

这里强调 filtered posterior，即只使用截至 $t$ 的信息。它不是 offline smoothing，不能使用未来价格或未来状态回看修正。

### 5.5 Production Stability Layer

原始 belief 仍可能受单个 bar 的噪声扰动。实盘不能让状态颜色跟着每个短期波动跳。稳定层使用 EWMA、hysteresis、minimum duration 和 transition gate，把 raw belief 转成稳定状态。

### 5.6 Responsiveness / Exit Pressure Layer

稳定性太强会导致滞后。因此系统又加入 responsiveness：当出现强 1h price shock、旧状态被连续证伪、或价格行为不再符合状态定义时，动态降低旧状态保留权。

### 5.7 Directional Quality 与 Price Expectation

市场结构成立不等于交易触发成立。HPEM 需要确认是否仍顺 PLIE 穿透；RHA 需要区分 reversal 与 stall；VT 需要区分 UP、DOWN、MIXED；RC、ST 也需要检查价格行为是否符合状态预期。

这些层输出交易质量与状态失效信号，帮助策略避免把低质量状态误用为交易机会。

---

## 6. 输入数据层：每类特征到底提供什么证据

本节不只列字段，而解释每类输入回答什么市场问题。

### 6.1 `base_context.csv`：主时钟与基础上下文

它回答：

```text
当前时点是什么？
当前价格是多少？
当前 split 是 train / validation / test 还是 live？
当前已有的 HMM / PLIE 基础状态是什么？
```

关键字段包括：

```text
time
price
split
hmm_state
hmm_conf
liq_entropy
age_in_state_source
state_severity
plie_direction
plie_main_bps
plie_reliability
plie_intensity
plie_phase
```

`base_context.csv` 也是 canonical frame 的主时钟来源。

### 6.2 `plie_predictions_source.csv`：当前清算压力源

PLIE 是 **Passive Liquidation Impact Estimate**，可以理解为“当前清算 forced flow 对未来短窗口价格可能施加的被动冲击基线”。

它回答：

```text
当前清算压力方向是什么？
强度多大？
是否可靠？
是否正在加速？
是否刚进入强清算状态？
```

关键字段包括：

```text
plie_main_bps
plie_passive_20m_bps
plie_passive_30m_bps
plie_passive_60m_bps
plie_direction
plie_reliability
plie_intensity
plie_accel_pos
plie_strong_entry
plie_transition_type
plie_transition_severity
```

PLIE 不是最终价格预测器。它只是 pressure source。最终状态必须看价格是否响应这股压力。

### 6.3 HMM liquidation regime：清算背景状态

HMM liquidation regime 回答：

```text
当前清算压力属于哪类背景状态？
压力方向是否明确？
置信度如何？
熵是否高？
当前状态已经持续多久？
```

关键字段：

```text
hmm_state
p_state_1 ... p_state_5
hmm_conf
liq_entropy
age_in_state_source
state_pressure_direction
state_severity
```

方向约定必须固定：

| HMM state | 金融含义 | 价格压力 |
|---:|---|---|
| 1 | 空头清算强势占优 | 向上压力强 |
| 2 | 空头清算轻度占优 | 向上压力轻 |
| 3 | 多空清算均衡 | 中性 |
| 4 | 多头清算轻度占优 | 向下压力轻 |
| 5 | 多头清算强势占优 | 向下压力强 |

### 6.4 `absorption_memory.csv`：成熟后的短期市场响应记忆

Market response memory 回答：

```text
最近已经成熟的清算事件之后，市场是传导、吸收、放大，还是反向接管？
```

它非常重要，因为 event-level response 需要未来价格才能计算，所以只能在响应窗口成熟后进入当前状态。模型不能使用当前事件尚未成熟的未来 response。

关键字段包括：

```text
mar_abs_score_q_staleaware_ewm_6_30m
mar_active_force_aligned_staleaware_ewm_6_30m
mar_active_force_price_staleaware_ewm_6_30m
mar_takeover_count_12_30m
mar_amplification_persistence_6_30m
mar_absorption_persistence_6_30m
mar_neutral_active_strength_evidence_ewm_6_30m
mar_response_conflict_score
```

它主要帮助区分：

```text
HPEM：短期放大持续；
RHA：短期吸收、stall 或 takeover；
VT：中性压力下主动运动强；
AMB：短期 response 与 path response 冲突。
```

### 6.5 `path_absorption_multiscale.csv`：多尺度路径吸收

这层回答：

```text
过去 6h / 12h / 24h / 48h 内，清算压力是什么结构？
价格路径如何响应这股压力？
短窗口和长窗口是否一致？
```

它是六状态识别的慢变量底盘。

关键字段包括：

```text
path_context_6h / 12h / 24h / 48h
path_label_6h / 12h / 24h / 48h
path_absorption_score_*
path_pressure_rejection_score_*
path_active_dominance_score_*
path_transmission_ratio_*
path_direction_consistency_*
path_cascade_score_*
path_data_quality_*
path_signal_clarity_*
path_activity_level_*
```

`path_context` 只描述压力本身，例如 directional、neutral、mixed；`path_label` 描述价格如何响应压力，例如 cascade、partial absorption、pressure rejection、active dominance、quiet no pressure。

### 6.6 `price_context_features.csv`：价格结构证据

这层回答：

```text
当前价格到底是安静、趋势、跳跃、压缩，还是高波动？
```

关键字段包括：

```text
past_return_1h / 3h / 6h / 12h / 24h
realized_vol_1h / 6h / 24h
range_compression_1h / 6h / 24h
trend_strength_1h / 6h / 24h
trend_consistency_1h / 6h / 24h
vol_of_vol_6h / 24h / 48h
jump_proxy_1h / 6h / 24h
price_gap_flag_*
price_missing_ratio_*
price_outlier_flag_*
```

这层不是为了直接预测收益，而是为了避免把低波误判为 RC、把高波误判为 HPEM、把主动趋势误判为 RHA。

### 6.7 `liqprice_features.csv` 与 `features_liq_dataflow.csv`：辅助市场结构与波动压力

当前 strict contract 要求它们必需提供，目的是保证每次运行使用同一套信息。它们提供辅助 price-liquidity 与 liquidation-volatility 结构，例如：

```text
trend_pressure
kalman_slope
vol_adaptive
fll_spike_kama
fsl_spike_kama
z_logtotalp
z_sdom
risk_priority_number
dominance
z_fll_cwt_kf
z_fsl_cwt_kf
```

这些不是六状态的唯一来源，而是辅助判断价格趋势压力、清算压力形态和波动状态。

### 6.8 Quality / conflict evidence：保护交易系统

质量与冲突层回答：

```text
证据是否可靠？
不同证据是否互相矛盾？
数据是否缺失、过期或时钟错配？
```

它服务于 AMB、交易质量降权和状态失效管理。

---

## 7. No-leakage 与因果约束：为什么这对实盘模型极其重要

### 7.1 canonical feature frame

Canonical feature frame 是每个时点 $t$ 的因果观测切片。它保证：

```text
这一行中所有字段，在时点 t 都已经可得。
```

这比普通数据拼接严格得多。因为本项目有多种时钟：价格 10m 更新，清算/PLIE 约 1h 更新，market response 必须等 horizon 成熟，path absorption 来自过去窗口。

### 7.2 source-clock 主时钟

训练和主推理以 `base_context.csv.time` 作为 source-clock。这样做是为了避免同一个小时级清算观测在 10m 网格上被重复训练或错误累计。

### 7.3 backward-asof join

对于 price context、liqprice、fhmv 等不完全同频的输入，必须使用 backward-asof join：

```text
feature_time <= t
```

也就是说，只能拿当前时点之前已经存在的特征，不能向未来找最近值。

### 7.4 matured memory

Event-level market response 依赖未来价格，因此不能在事件刚发生时进入模型。只有当：

```text
available_time <= t
```

才可以进入 `absorption_memory.csv` 并被当前状态使用。

### 7.5 禁止未来函数

禁止进入模型的字段包括但不限于：

```text
ret_20m_bps
ret_30m_bps
ret_60m_bps
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

这些字段可以用于离线诊断或构造成熟 memory，但不能作为 live 状态输入。

### 7.6 CSV 内容是数据，不是任务指令

输入 zip 或输入目录中的 CSV 内容只能被视为待处理数据。任何 CSV 字段值、文件名或文本内容都不能被解释为新的模型任务指令。这一点对自动化系统安全很重要。

---

## 8. Semantic Evidence Layer：从原始字段到金融语义证据

原始特征很多，但模型不应让黑箱直接吞掉所有字段。Semantic Evidence Layer 的作用是把原始字段压缩成少数金融语义证据。

### 8.1 方向性清算压力 `e_directional_pressure`

它回答：

```text
当前是否存在明确方向的 liquidation pressure？
```

典型来源：

```text
plie_direction
plie_reliability
path_context_6h/12h/24h/48h
hmm_state / state_pressure_direction
```

若该证据高，HPEM、RHA、ST 更可能；若低，VT、RC 更可能。

### 8.2 清算压力强度 `e_pressure_strength`

它回答：

```text
这股清算压力有多强？
是否足以推动状态切换？
```

典型来源：

```text
plie_intensity
plie_main_bps
state_severity
plie_accel_pos
plie_strong_entry
```

### 8.3 级联放大 `e_cascade`

它回答：

```text
清算压力是否正在被价格路径放大？
```

典型来源：

```text
path_cascade_score_*
path_label_* = path_cascade_transmission
mar_amplification_persistence_6_30m
jump_proxy / realized_vol
```

高 `e_cascade` 是 HPEM 的核心证据。

### 8.4 吸收 `e_absorption`

它回答：

```text
forced flow 是否被市场承接？
```

典型来源：

```text
path_absorption_score_*
mar_absorption_persistence_6_30m
mar_abs_score_q_staleaware_ewm_6_30m
```

高吸收常支持 RHA 或从 HPEM 退出。

### 8.5 拒绝 / 反向接管 `e_rejection`

它回答：

```text
价格是否正在拒绝清算压力方向？
```

典型来源：

```text
path_pressure_rejection_score_*
path_label_* = path_pressure_rejection / path_reversal_takeover
mar_takeover_count_12_30m
```

这是 RHA 的核心证据。

### 8.6 主动主导 `e_active`

它回答：

```text
价格是否主要由主动交易或外生冲击推动，而不是 PLIE？
```

典型来源：

```text
path_active_dominance_score_*
path_label_* = path_active_dominance_up/down
mar_neutral_active_strength_evidence_ewm_6_30m
trend_strength / realized_vol
```

这是 VT 的核心证据。

### 8.7 安静 / 区间 `e_quiet`, `e_range_compression`, `e_low_trend`

它回答：

```text
市场是否缺少压力、缺少趋势、价格区间是否压缩？
```

这些证据支持 RC，但如果 trend_strength 或 range break 上升，则 RC 应被削弱。

### 8.8 冲突与低质量 `e_conflict`, `e_quality_bad`

它回答：

```text
模型现在是否应该保护性降权或进入 AMB？
```

典型来源：

```text
mar_response_conflict_score
liq_entropy
hmm_conf 低
path_signal_clarity 低
path_data_quality 低
price_gap_flag / missing_rate
belief_gap 低
```

---

## 9. 六状态打分与 emission prior

Semantic evidence 构造完成后，模型会为每个状态生成一个语义分数。这个分数不是最终状态，而是 emission prior 的基础。

一般形式为：

$$
s_t^k = w_k^\top e_t + u_k^\top c_t + b_k,
$$

其中：

```text
k       表示状态 ST / VT / RC / RHA / HPEM / AMB；
e_t     表示连续 evidence 向量；
c_t     表示类别证据，如 path_label、path_context、plie_phase；
w_k,u_k 表示该状态对不同证据的权重；
b_k     表示状态偏置。
```

直觉上，每个状态都有自己的“证据配方”。

### 9.1 HPEM score 的直觉

HPEM 需要：

```text
方向性压力强；
级联放大强；
价格波动 / 跳跃强；
吸收低；
冲突低。
```

可写成：

$$
s_t^{HPEM}
\propto
+e_{directional}
+e_{pressure\_strength}
+e_{cascade}
+e_{volatility}
+e_{jump}
-e_{absorption}
-e_{conflict}.
$$

### 9.2 RHA score 的直觉

RHA 需要：

```text
清算压力仍在；
价格拒绝压力；
吸收 / takeover 上升；
不能是 neutral active trend。
```

$$
s_t^{RHA}
\propto
+e_{directional}
+e_{absorption}
+e_{rejection}
+e_{takeover}
-e_{neutral\_pressure}
-e_{cascade}.
$$

### 9.3 VT score 的直觉

VT 需要：

```text
主动主导；
高波动或趋势强；
清算解释力弱或中性；
不能只是清算级联。
```

$$
s_t^{VT}
\propto
+e_{active}
+e_{trend\_strength}
+e_{volatility}
+e_{neutral\_or\_mixed}
-e_{strong\_directional\_PLIE}.
$$

### 9.4 RC score 的直觉

RC 需要：

```text
低压力；
低趋势；
低波动；
区间压缩；
低冲突。
```

$$
s_t^{RC}
\propto
+e_{quiet}
+e_{low\_vol}
+e_{range\_compression}
+e_{low\_trend}
-e_{active}
-e_{conflict}
-e_{jump}.
$$

### 9.5 AMB score 的直觉

AMB 需要：

```text
证据冲突；
数据质量问题；
低 clarity；
belief gap 低。
```

$$
s_t^{AMB}
\propto
+e_{conflict}
+e_{quality\_bad}
+e_{mixed\_chop}
+e_{high\_entropy}.
$$

### 9.6 ST score 的直觉

ST 需要：

```text
有序趋势；
baseline / partial transmission；
中等压力；
中等波动；
冲突低。
```

$$
s_t^{ST}
\propto
+e_{baseline}
+e_{orderly\_trend}
+e_{moderate\_pressure}
+e_{moderate\_vol}
-e_{cascade}
-e_{rejection}
-e_{conflict}.
$$

### 9.7 softmax emission prior

状态分数通过 softmax 变成先验概率：

$$
\psi_t(k)=
\frac{\exp(s_t^k/\tau)}{\sum_j \exp(s_t^j/\tau)}.
$$

其中 $\tau$ 是温度参数。直觉上，$\tau$ 越小，模型越相信最高分状态；$\tau$ 越大，模型越保留不确定性。

---

## 10. CD-IOHSMM-lite：为什么状态切换需要路径和持续时间

### 10.1 普通 HMM 的不足

普通 HMM 可以表达隐状态与观测之间的关系，但它通常存在两个问题：

```text
1. 状态持续时间隐含为几何分布，容易不符合真实市场状态；
2. 转移矩阵固定，不能根据当前金融证据动态决定是否切换。
```

BTC 市场状态切换必须有金融证据。例如 HPEM -> RHA 应该由 absorption / rejection 增强支持，而不是随机跳转。

### 10.2 duration-aware 的意义

模型维护状态年龄 $a_t$：

$$
a_t =
\begin{cases}
 a_{t-1}+1, & z_t=z_{t-1},\\
 1, & z_t\ne z_{t-1}.
\end{cases}
$$

这个公式的直觉是：状态持续越久，退出条件可能不同。例如 HPEM 初期可以快速发展，但持续很久后，如果放大衰减且吸收上升，就更应该退出。

### 10.3 exit hazard

退出概率使用 state-specific hazard：

$$
h_i(a_{t-1},u_t)
=
\sigma\left(\gamma_{i,0}+\gamma_{i,a}\log(1+a_{t-1})+\gamma_i^\top u_t\right).
$$

直觉上：

```text
基础退出倾向 + 状态年龄 + 当前退出证据 → 当前状态是否应该退出。
```

其中 $u_t$ 可以包含：

```text
short price shock
absorption/rejection 上升
cascade 衰减
quality/conflict 上升
price expectation fail
```

### 10.4 transition probability

如果状态 $i$ 退出，转向状态 $j$ 的概率为：

$$
A_{ij}(u_t)=
\frac{\exp(\beta_{ij,0}+\beta_{ij}^\top u_t+M_{ij}(u_t))}
{\sum_{l\ne i}\exp(\beta_{il,0}+\beta_{il}^\top u_t+M_{il}(u_t))}.
$$

$M_{ij}$ 是金融语义 transition mask。例如：

```text
HPEM -> RHA 需要 absorption / rejection；
RC -> VT 需要 active dominance / trend breakout；
ST -> HPEM 需要 cascade / PLIE acceleration；
任意状态 -> AMB 需要 conflict / quality bad。
```

### 10.5 filtered posterior

在线过滤递推维护：

$$
\alpha_t(k,a)=P(z_t=k,a_t=a,x_{1:t}).
$$

输出 belief：

$$
b_t^k=\sum_a \alpha_t(k,a).
$$

直觉上，模型不是事后回看整段行情再判断，而是每个时点只用当前和过去证据更新状态。这就是 filtered posterior。它与 offline smoothed posterior 的区别非常重要：后者使用未来信息，不可用于 live trading。

---

## 11. Production Stability：为什么不能让状态被单根 bar 随意扰动

### 11.1 为什么需要稳定层

raw belief 可能会因为单根价格冲击或短期噪声出现波动。实盘系统如果直接使用 `argmax(b_t)`，可能导致状态颜色频繁跳变，交易策略也会过度换手。

Production stability layer 的目标是：

```text
保留对真实状态变化的响应；
过滤单根噪声；
让状态切换具备持续性和金融解释。
```

### 11.2 EWMA stable belief

稳定 belief 定义为：

$$
\tilde b_t = \lambda \tilde b_{t-1} + (1-\lambda)b_t,
\qquad 0<\lambda<1.
$$

直觉上，$\lambda$ 控制惯性。$\lambda$ 越大，状态越稳定，但响应越慢。

### 11.3 hysteresis 与 minimum duration

进入新状态通常需要满足：

```text
candidate belief 超过 tau_in；
belief_gap 足够大；
旧状态持续时间超过 minimum duration；
目标状态 gate 通过；
数据质量没有阻断。
```

这可以防止状态被单点噪声触发。

### 11.4 exit pressure

稳定过强会导致迟缓。因此系统引入 exit pressure memory。若旧状态连续被证伪，退出压力会累积：

$$
X_t = \rho X_{t-1} + (1-\rho)I_t,
$$

其中 $I_t$ 是当前状态失效证据，$X_t$ 是累计退出压力。

直觉上，一个反向 bar 不足以退出；连续的反向证据会逐步降低旧状态保留权。

---

## 12. Directional Quality：为什么有状态还不够，还要判断交易方向

### 12.1 stable_state 不等于 trade signal

市场结构成立，不代表可以直接交易。例如：

```text
stable_state = RHA
```

可能表示价格明确反向接管，也可能只是清算压力被吸收但价格横住。前者可能有交易价值，后者更适合观察。

所以系统增加 directional quality layer。

### 12.2 HPEM_UP / HPEM_DOWN

HPEM 的交易方向应与 PLIE 方向一致。若 PLIE 方向为 $d^{PLIE}$，则状态段内 aligned move 为：

$$
m_t^{HPEM}=d^{PLIE}\cdot 10000\log\left(\frac{P_t}{P_{entry}}\right).
$$

直觉上，它衡量进入 HPEM 后，价格是否顺清算压力方向移动。

段内最大有利移动：

$$
MFE_t^{HPEM}=\max_{u\le t} m_u^{HPEM}.
$$

回吐：

$$
Giveback_t^{HPEM}=MFE_t^{HPEM}-m_t^{HPEM}.
$$

如果 HPEM 初期顺 PLIE 移动，但随后大幅回吐，说明 HPEM 的穿透含义正在失效。

### 12.3 RHA_REVERSAL / RHA_STALL / INVALID_RHA

RHA 的方向应与 PLIE 相反。定义：

$$
m_t^{RHA}=(-d^{PLIE})\cdot 10000\log\left(\frac{P_t}{P_{entry}}\right).
$$

如果 $m_t^{RHA}$ 持续上升，说明反向接管成立，可输出 `RHA_REVERSAL_UP/DOWN`。如果价格只是横住，输出 `RHA_STALL`。如果价格重新服从 PLIE，则输出 `INVALID_RHA_FOLLOWS_PLIE`。

### 12.4 VT_UP / VT_DOWN / VT_MIXED

VT 的方向不能依赖 PLIE，因为 VT 的定义就是 PLIE 不是主解释变量。VT 方向来自价格结构和 active dominance。

方向分数可以写为：

$$
D_t^{VT}
=
0.25\,d_{6h}^{trend}
+0.20\,\operatorname{sign}(r_{3h})
+0.15\,d_{1h}^{trend}
+0.20\,d_{6h}^{active}
+0.10\,d_{12h}^{active}
+0.10\,d_{24h}^{active}.
$$

直觉上，它综合短中期价格方向与 path active dominance。如果分数明显为正，输出 `VT_UP`；明显为负，输出 `VT_DOWN`；方向混合则输出 `VT_MIXED`。

---

## 13. Price Expectation Compliance：状态成立后，价格行为还必须继续符合定义

### 13.1 为什么需要 price expectation layer

即使 stable_state 已经成立，状态段内价格行为也可能逐渐不再符合该状态定义。例如：

```text
HPEM 初期顺 PLIE 级联，但后续大幅回吐；
RHA 初期吸收，但随后价格重新服从 PLIE；
RC 内价格漂移过大；
VT 进入后价格位移不足；
ST 变成横盘或极端跳跃。
```

Price expectation compliance layer 持续检查：当前状态段内价格行为是否仍符合该状态的金融定义。

### 13.2 HPEM 的价格预期

HPEM 应顺 PLIE 方向连续穿透。如果出现：

```text
no response：进入后没有顺 PLIE 位移；
adverse：价格明显逆 PLIE；
decay：先顺 PLIE 后大幅回吐；
```

则生成 `HPEM_PRICE_EXPECTATION_FAIL`，降低交易质量，并可能触发 `price_expectation_exit_flag`。

### 13.3 RHA 的价格预期

RHA 作为交易触发时，应表现为反 PLIE 的 reversal。若只是横住，属于 `RHA_STALL`，状态语义可以保留但交易质量应降低。若价格重新服从 PLIE，则是 `RHA_INVALID_FOLLOWS_PLIE`，应提高退出压力。

### 13.4 RC 的价格预期

RC 应价格变化很小。如果 entry-to-current return、range 或 trend_strength 超过阈值，说明 RC 价格行为不再符合整理，应退出到 VT、ST 或其他更合适状态。

### 13.5 VT 的价格预期

VT 价格变化应较大，方向可以短期变化。如果价格变化很小或波动不足，则应标记为 `VT_LOW_MOVE` 或 `VT_WEAK_VOLATILITY`，不应作为高质量交易触发。

### 13.6 ST 的价格预期

ST 应有温和有序趋势。如果价格几乎不动，是 `ST_FLAT`；如果极端跳跃，则可能不再是 ST，而更接近 VT/HPEM 或 AMB。

---

## 14. 输出字段如何被策略使用

### 14.1 核心输出

| 字段 | 用途 |
|---|---|
| `b_ST ... b_AMB` | 六状态 belief vector。 |
| `candidate_state` | raw / filtered belief 的即时候选状态。 |
| `stable_state` | 稳定层后的市场结构状态。 |
| `belief_gap` | 状态置信边际。 |
| `directional_state` | 交易方向解释，如 `HPEM_DOWN`, `VT_MIXED`。 |
| `trade_direction` | `UP / DOWN / MIXED / NO_TRADE`。 |
| `direction_confidence` | 方向置信度。 |
| `directional_trading_quality` | 方向交易质量。 |
| `state_price_expectation_score` | 状态内价格行为是否符合定义。 |
| `price_expectation_exit_flag` | 当前状态是否出现价格行为失效。 |
| `price_expectation_exit_target` | 建议关注的退出目标状态。 |
| `transition_reason` | 状态切换解释。 |
| `top_evidence_features` | 主要证据。 |

### 14.2 推荐策略读取方式

策略应将输出分层使用：

```text
stable_state:
    用作市场结构语境。

directional_state:
    用作方向解释。

trading_trigger_quality:
    用作交易触发过滤。

state_price_expectation_score:
    用作状态内行为合规检查。

price_expectation_exit_flag:
    用作减仓、退出或停止加仓提醒。

belief_gap:
    用作状态置信边际。
```

示例：

```python
if stable_state == "HPEM":
    if directional_state in {"HPEM_UP", "HPEM_DOWN"} \
       and directional_trading_quality >= q_min \
       and state_price_expectation_score >= pe_min \
       and price_expectation_exit_flag == 0 \
       and belief_gap >= gap_min:
        allow_trend_follow_context()
    else:
        observe_or_reduce_risk()
```

### 14.3 状态段诊断输出

系统还输出 segment-level 文件，例如：

```text
segment_diagnostics.csv
segment_quality_flags.csv
low_quality_segment_examples.csv
segment_quality_summary.json
```

这些文件用于审查：

```text
短状态段是否过多；
HPEM 是否顺 PLIE；
RHA 是否是 reversal 还是 stall；
RC 是否价格变化过大；
VT 是否低位移；
ST 是否缺少趋势。
```

---

## 15. 如何评价这个没有人工标签的状态模型

由于没有人工六状态标签，不能用 accuracy 作为主指标。评价要围绕金融一致性、状态段质量和交易相关价值。

### 15.1 状态金融一致性

检查每个状态的证据分布是否符合定义：

```text
HPEM 是否 high PLIE + cascade + high vol/jump？
RHA 是否 high absorption/rejection/takeover？
VT 是否 active dominance + low PLIE explanation？
RC 是否 low PLIE + quiet + low vol + compression？
AMB 是否 high conflict / low quality / mixed pressure？
ST 是否 baseline / orderly trend / moderate vol？
```

### 15.2 状态段质量

逐段检查：

```text
duration_bar_count
entry_price / exit_price
segment_return_bps
segment_range_bps
max favorable / adverse move
mean realized vol
trend_strength / trend_consistency
jump_proxy
path / PLIE evidence
state_quality_issues
```

### 15.3 切换解释

每次切换应输出：

```text
from_state
to_state
transition_probability
top evidence
path evidence
market_response evidence
PLIE evidence
duration evidence
quality/conflict evidence
```

### 15.4 OOS 稳定性

用时间切分或 walk-forward 检查：

```text
state distribution stability
transition matrix stability
state semantic profile stability
duration distribution stability
feature drift
belief entropy / gap drift
```

### 15.5 交易相关评价

最终应验证：

```text
每个状态下策略表现；
状态切换后的风险和收益；
AMB 是否减少低质量交易；
RHA 是否提示反转/减仓；
HPEM 是否提示风险放大；
price_expectation_exit_flag 是否改善退出风险。
```

---

## 16. 常见误用与风险

### 16.1 把 HPEM 当作无条件追涨杀跌

错误。HPEM 必须检查方向质量与 price expectation。`WEAK_HPEM` 或 `HPEM_PRICE_EXPECTATION_FAIL` 不应作为高质量追随信号。

### 16.2 把所有 RHA 当作反转交易

错误。RHA 包含 reversal 和 stall。只有 `RHA_REVERSAL_UP/DOWN` 更接近方向交易；`RHA_STALL` 主要表示吸收，不一定适合开方向仓。

### 16.3 把 RC 当作一定可做区间

错误。RC 只是低压力、低趋势、低波动语境。做区间策略还需要边界、成交、成本和风险管理确认。

### 16.4 把 VT 当作固定方向

错误。VT 可以是 `VT_UP`、`VT_DOWN` 或 `VT_MIXED`。如果是 `VT_MIXED`，应降低方向交易权重。

### 16.5 只看 stable_state

错误。策略至少应同时检查：

```text
belief_gap
directional_state
trading_trigger_quality
state_price_expectation_score
price_expectation_exit_flag
```

### 16.6 忽略 no-leakage

严重错误。任何未来收益、未成熟 response 或 forward-filled feature 都会让状态结果在回测中虚假变好，但无法实盘复现。

---

## 17. 当前实现边界与未来升级

### 17.1 当前系统已经实现

当前系统已经实现：

```text
strict fixed input bundle；
feature contract validation；
no-leakage canonical frame；
semantic evidence layer；
CD-IOHSMM-lite filtered belief engine；
production stability layer；
responsiveness / exit pressure；
causal state confirmation；
directional quality layer；
price expectation compliance；
segment diagnostics；
interactive dashboard。
```

### 17.2 当前系统不是什么

它不是：

```text
收益预测模型；
普通六分类器；
无解释聚类；
自动买卖信号；
保证收益的策略；
可以忽略交易成本和执行风险的信号生成器。
```

### 17.3 未来可能升级

后续可以考虑：

```text
更完整的 constrained EM；
posterior regularization；
更严格的 transition learning；
factorial latent state decomposition；
不同交易所 / 现货 / 永续 / ETF 载体的层次建模；
策略层根据状态输出做成本后风险收益验证。
```

---

## 18. 总结：市场机制底盘，而不是单一买卖信号

本模型应该被理解为一个市场机制底盘。它把 BTC 杠杆市场中的清算压力、价格响应、吸收/传导、主动交易、波动结构和证据冲突整合成六状态 belief 与稳定状态。

最重要的使用原则是：

```text
stable_state                  理解市场结构；
directional_state             理解交易方向；
trading_trigger_quality       判断是否值得交易；
state_price_expectation_score 判断状态内价格行为是否仍符合定义；
price_expectation_exit_flag   管理状态失效、减仓或退出风险；
belief_gap                    判断状态置信边际。
```

它不会替代策略设计、仓位管理、成本模型和风险控制。但它能为这些模块提供更清晰的市场状态语境，帮助策略避免把清算压力、主动趋势、高吸收、低波整理和证据冲突混为一谈。

一句话概括：

> 这不是一个告诉你“买还是卖”的模型，而是一个告诉你“市场当前由什么机制主导、这个机制是否清晰、是否仍然有效、是否适合交易”的状态推理引擎。

---

## 附录 A：核心输入文件清单

当前版本采用 strict fixed input bundle，以下文件均为必需：

```text
base_context.csv
plie_predictions_source.csv
absorption_memory.csv
path_absorption_multiscale.csv
path_absorption.csv
price_context_features.csv
liqprice_features.csv
features_liq_dataflow.csv
```

如果缺少必需文件或必需列，pipeline 应报错，不应静默降级。

---

## 附录 B：核心输出字段清单

```text
stable_state
candidate_state
b_ST, b_VT, b_RC, b_RHA, b_HPEM, b_AMB
belief_gap
state_duration
transition_reason
top_evidence_features

directional_state
trade_direction
direction_confidence
directional_trading_quality
directional_no_trade_reason

state_confirmation_score
state_confirmation_status
trading_trigger_quality
weak_state_flag

state_price_expectation_score
state_price_expectation_issue
price_expectation_exit_flag
price_expectation_exit_target
price_expectation_exit_reason

segment_id
bars_since_state_entry
entry_to_current_return_bps
entry_to_current_range_bps
```

---

## 附录 C：论文方法论表述建议

如果要在论文或研究报告中描述本模型，可以使用以下表述：

> We model BTC liquidation-regime dynamics as a causal, path-dependent, semantically constrained latent state inference problem. The model does not forecast future returns directly. Instead, it infers a six-dimensional belief vector over interpretable market mechanisms, using liquidation pressure, PLIE, matured market response memory, multiscale path absorption, price context, and quality/conflict evidence. A duration-aware online filtering engine produces raw posterior beliefs, which are then stabilized by hysteresis, minimum duration, transition constraints, responsiveness, directional quality, and price expectation compliance layers. The final output is intended as a market-mechanism substrate for strategy conditioning and risk management, not as a standalone trading signal.

中文版本：

> 本文将 BTC 清算市场状态建模为一个严格因果、路径依赖、金融语义约束的 latent state inference problem。模型不直接预测未来收益，而是基于清算压力、PLIE、成熟 market response memory、多尺度 path absorption、价格上下文与质量/冲突证据，推断六个可解释市场机制上的 belief vector。在线 filtered belief 经过 duration-aware 推理、hysteresis、minimum duration、transition constraints、responsiveness、directional quality 与 price expectation compliance 处理后，形成稳定市场状态与交易质量输出。该输出用于策略条件化与风险控制，而不是单独作为买卖信号。
