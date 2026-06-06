# V2.4 Directional Quality Layer

本版本在既有六状态 `stable_state` 后增加方向生命周期层，不改变六状态定义，也不覆盖 filtered belief / stable state。

## 目标

1. 检查 HPEM 是否仍然顺 PLIE 压力方向穿透。
2. 检查 HPEM 是否发生 MFE 回吐 / giveback decay。
3. 将 RHA 拆成交易层子类型：`RHA_REVERSAL_UP/DOWN`、`RHA_STALL`、`RHA_WEAK`、`INVALID_RHA_FOLLOWS_PLIE`。
4. 将 VT 拆成交易方向层：`VT_UP`、`VT_DOWN`、`VT_MIXED`。
5. 输出 `directional_trading_quality` 与 `directional_no_trade_reason`，用于交易触发过滤。

## 因果约束

所有方向生命周期指标均基于当前状态段从进入时点到当前时点的已发生路径计算，例如：

- `entry_to_current_return_bps`
- `hpem_aligned_mfe_bps`
- `hpem_giveback_bps`
- `rha_reversal_mfe_bps`
- `vt_direction_score`

不会使用状态段未来终点或未来价格回填当前标签。

## 新增输出字段

### 通用字段

```text
directional_state
trade_direction
direction_confidence
direction_source
directional_trading_quality
directional_no_trade_reason
directional_exit_trigger_flag
directional_exit_reason
directional_exit_target
```

### HPEM

```text
hpem_expected_direction
hpem_aligned_move_bps
hpem_aligned_mfe_bps
hpem_giveback_bps
hpem_giveback_ratio
hpem_directional_compliance_score
hpem_no_response_flag
hpem_decay_flag
hpem_adverse_flag
```

### RHA

```text
rha_expected_direction
rha_reversal_move_bps
rha_reversal_mfe_bps
rha_follow_plie_move_bps
rha_reversal_score
rha_directional_compliance_score
rha_stall_flag
rha_reversal_confirmed_flag
rha_invalid_follow_plie_flag
rha_subtype
```

### VT

```text
vt_direction_score
vt_direction_state
vt_direction_confidence
vt_up_ratio_recent
vt_down_ratio_recent
vt_direction_flip_flag
```

## 交易解释

`stable_state` 表示市场结构状态；`directional_state` 表示交易方向解释层。例如：

```text
stable_state = VT
方向层可能为：VT_UP / VT_DOWN / VT_MIXED
```

```text
stable_state = HPEM
方向层可能为：HPEM_UP / HPEM_DOWN / WEAK_HPEM
```

```text
stable_state = RHA
方向层可能为：RHA_REVERSAL_UP / RHA_REVERSAL_DOWN / RHA_STALL / RHA_WEAK
```

交易模块应优先使用 `directional_trading_quality`、`directional_state` 与 `trade_direction`，而不是只使用 `stable_state`。

## 回滚方式

在 `configs/model_config.yaml` 中设置：

```yaml
directional_quality:
  enabled: false
```

即可关闭本层。
