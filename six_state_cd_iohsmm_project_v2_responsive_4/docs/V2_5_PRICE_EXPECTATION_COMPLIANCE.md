# V2.5 Price Expectation Compliance Layer

## 1. Purpose

V2.5 adds a causal price-behavior compliance layer on top of the existing six-state engine.
It does **not** redefine ST / VT / RC / RHA / HPEM / AMB, and it does not replace filtered belief or stable_state.
It continuously checks whether the price path since the current stable-state entry still matches the expected price behavior of that state.

The layer was added to address these review findings:

- HPEM should show directional continuation in the PLIE pressure direction; if it loses continuity, gives back, or moves against PLIE, the state should lose trading quality and eventually exit.
- RHA should distinguish reversal from stall. Reversal has directional trading value; stall has state meaning but should be low/no-trade quality. If price follows PLIE again, RHA is invalidated.
- RC should have small price movement, low trend, and range compression. If the segment drifts or breaks range, RC should be downgraded and exit pressure should rise.
- VT should have large movement or high active/volatile behavior. Direction may flip, but low-move VT is weak and VT_MIXED is no-trade for direction.
- ST should be mild orderly trend. Flat ST and extreme ST are downgraded.

## 2. New row-level outputs

The layer writes:

```text
state_price_expectation_score
state_price_expectation_issue
price_expectation_exit_flag
price_expectation_exit_target
price_expectation_exit_reason
price_expectation_exit_pressure
price_expectation_trading_quality
price_expectation_raw_issue_flag

rc_price_drift_flag
rc_range_break_flag
rc_trend_exclusion_flag

vt_low_move_flag
vt_mixed_direction_flag
vt_weak_trend_flag

st_flat_flag
st_weak_trend_flag
st_too_extreme_flag

hpem_price_expectation_fail_flag
rha_price_expectation_fail_flag
```

The layer also updates:

```text
trading_trigger_quality
directional_trading_quality
directional_no_trade_reason
```

by taking the minimum between the existing quality and price-expectation quality.

## 3. HPEM rules

HPEM is expected to continue in PLIE pressure direction.

Inputs used include:

```text
hpem_no_response_flag
hpem_decay_flag
hpem_adverse_flag
hpem_directional_compliance_score
hpem_price_impact_score
hpem_giveback_ratio
```

A failed HPEM receives:

```text
state_price_expectation_issue = HPEM_PRICE_EXPECTATION_FAIL
price_expectation_exit_flag = 1
price_expectation_exit_target = RHA / VT / ST / AMB, depending on evidence
```

## 4. RHA rules

RHA is separated into tradeable reversal and non-trade stall/weak cases.

Inputs used include:

```text
rha_invalid_follow_plie_flag
rha_stall_flag
rha_reversal_confirmed_flag
rha_directional_compliance_score
```

Outcomes:

```text
RHA_INVALID_FOLLOWS_PLIE -> hard exit suggestion
RHA_STALL_NOT_DIRECTIONAL -> low trading quality, stable state may remain RHA
RHA_WEAK_NO_REVERSAL -> low trading quality
```

## 5. RC rules

RC should have small price movement and low trend.

Failure types:

```text
RC_PRICE_DRIFT
RC_RANGE_BREAK
RC_TREND_EXCLUSION
```

When any is triggered, the layer emits:

```text
state_price_expectation_issue = RC_PRICE_TOO_LARGE_OR_TRENDING
price_expectation_exit_flag = 1
```

## 6. VT rules

VT should have large movement or sufficient volatility/active trend. Direction may flip, but mixed direction should not be a directional trading trigger.

Failure/low-quality types:

```text
VT_LOW_MOVE
VT_WEAK_VOLATILITY
VT_MIXED_DIRECTION
```

## 7. ST rules

ST should be a mild orderly trend.

Failure types:

```text
ST_FLAT
ST_WEAK_TREND
ST_TOO_EXTREME
```

## 8. Stability-layer integration

The `ProductionStabilityLayer` can consume price-expectation invalidity through:

```yaml
state_price_expectation:
  enabled: true
  use_in_stability: true
  stability_boost:
    soft_threshold: 0.55
    hard_threshold: 0.75
    tau_discount: ...
```

This allows strong failures to add exit pressure, reduce effective thresholds, and select a semantically gated target state.

## 9. Causality

All computations are causal. Entry-to-current returns, ranges, MFE/MAE and compliance scores only use observations that have occurred since the current stable-state entry up to the current row. The layer does not use future segment end information to rewrite past labels.

## 10. Recommended use

Use:

```text
stable_state
```

for market context, and use:

```text
execution_state
execution_quality
state_price_expectation_score
price_expectation_exit_flag
```

for trading trigger control.

A simple gate is:

```text
if execution_quality < 0.45:
    no directional trade
if price_expectation_exit_flag == 1:
    treat current stable state's trading meaning as decaying or invalid
```
