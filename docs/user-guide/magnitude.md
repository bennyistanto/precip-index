# Understanding Event Magnitude: Cumulative vs Instantaneous

**Date:** 2026-01-21
**Version:** 2026.1

> **Note:** This guide uses drought (dry event) terminology in examples for clarity, but all concepts apply equally to wet events (flooding/excess precipitation). Simply reverse the threshold direction: negative thresholds (< 0) identify dry events, positive thresholds (> 0) identify wet events.

---

## Overview

The `calculate_timeseries()` function provides **TWO types of magnitude** to support different analysis needs for **both dry and wet climate extremes**:

1. **Magnitude (Cumulative)** - Total accumulated deviation from threshold
2. **Magnitude (Instantaneous)** - Current month's severity

Both are valid and useful - they measure different aspects of event evolution.

---

## 1. Magnitude (Cumulative)

### Definition

Sum of all monthly deviations from the threshold, accumulated throughout the event.

### Formula

```
magnitude_cumulative[t] = Σ|threshold - SPI[i]| for all i from event start to time t
```

### Behavior

- **Always increases** during event (never decreases within an event)
- Resets to 0 when event ends
- Represents **total accumulated deviation** from threshold

### Analogy: Debt Accumulation

Think of it like credit card debt:

- Each month you spend beyond budget → debt increases
- Total debt = sum of all monthly overspending
- Debt never decreases until you pay it off (event ends)

### Use Cases

✅ **Total impact assessment**: How much total deviation occurred?   
✅ **Event comparison**: Which event had greater total magnitude?   
✅ **Resource planning**: Total deviation to recover from   
✅ **Impact estimation**: Cumulative stress (water deficit for dry, excess for wet)   
✅ **Period statistics**: Total magnitude within time range

### Example Pattern (Dry Event)

```
Month:  1     2     3     4     5     6     7    (event ends)
SPI:   -1.5  -2.0  -1.8  -1.3  -1.1  -0.8   0.5
Mag:    0.3   0.8   1.4   1.7   1.9   2.1   0.0  ← Always increasing!
```

---

## 2. Magnitude (Instantaneous)

### Definition

Current month's deviation from the event threshold. Equal to the monthly deficit.

### Formula

```
magnitude_instantaneous[t] = threshold - SPI[t]  (if SPI < threshold)
                           = 0                    (if SPI >= threshold)
```

### Behavior

- **Varies with SPI** (rises when event worsens, falls when event eases)
- Shows event evolution: intensification → peak → recovery
- Represents **current severity** at this moment

### Analogy: Crop NDVI Phenology

Think of it like crop greenness (NDVI) over a growing season:

- Early season: NDVI rises (crop growing)
- Mid-season: NDVI peaks (maximum greenness)
- Late season: NDVI falls (approaching harvest)
- Pattern: **rise → peak → fall**

For event severity (instantaneous magnitude):

- Early event: severity rises (worsening conditions)
- Peak event: severity at maximum
- Late event: severity falls (easing conditions)
- Pattern: **rise → peak → fall**

### Use Cases

✅ **Monitoring event evolution**: Is it getting worse or better?
✅ **Identifying peak severity**: When was the worst moment?
✅ **Real-time tracking**: Current stress level right now
✅ **Early warning**: Severity increasing = event intensifying
✅ **Recovery monitoring**: Severity decreasing = event easing

### Example Pattern

```
Month:  1     2     3     4     5     6     7    (event ends)
SPI:   -1.5  -2.0  -1.8  -1.3  -1.1  -0.8   0.5
Mag:    0.3   0.8   0.6   0.1   0.0   0.0   0.0  ← Rises, peaks, falls!
```

---

## Side-by-Side Comparison

### Same Event, Different Perspectives

```
Time:           1     2     3     4     5     6     7
SPI:          -1.5  -2.0  -1.8  -1.3  -1.1  -0.8   0.5
Threshold:    -1.2  -1.2  -1.2  -1.2  -1.2  -1.2  -1.2

Cumulative:    0.3   1.1   1.7   1.8   1.9   1.9   0.0  ← Total debt
Instantaneous: 0.3   0.8   0.6   0.1   0.0   0.0   0.0  ← Current rate
```

**Interpretation:**

- **Month 2**: Event worsening (instantaneous peaks at 0.8)
- **Month 3**: Event easing (instantaneous falls to 0.6)
- **Month 6**: Near recovery (instantaneous = 0, but cumulative still high)
- **Total impact**: Cumulative magnitude = 1.9 (total water deficit)

---

## Visualization Differences

### Plot 1: Cumulative Magnitude

```
 2.0 ┤                           ╭─────╮
 1.5 ┤                   ╭───────╯     │
 1.0 ┤           ╭───────╯             │
 0.5 ┤   ╭───────╯                     │
 0.0 ┼───╯                             ╰────
     └─────────────────────────────────────
     Pattern: Staircase up, then drops to 0
     Interpretation: Total deficit accumulating
```

### Plot 2: Instantaneous Magnitude

```
 0.8 ┤       ╭╮
 0.6 ┤      ╱  ╲
 0.4 ┤    ╱      ╲
 0.2 ┤  ╱          ╲___
 0.0 ┼─╯                ╲─────────────────
     └─────────────────────────────────────
     Pattern: Rise → Peak → Fall (like NDVI)
     Interpretation: Severity varies with SPI
```

---

## When to Use Each?

### Use Cumulative Magnitude When:

- Calculating **total water deficit** for the event
- Comparing **overall severity** between events
- Estimating **agricultural losses** (total stress)
- Planning **water resource needs** (total deficit to recover)
- Creating **period statistics** (total magnitude in 2023)
- Answering: *"How much total deficit occurred?"*

### Use Instantaneous Magnitude When:

- Monitoring **current dry/wet conditions**
- Tracking **event evolution** (worsening or easing?)
- Identifying **peak severity** timing
- **Early warning systems** (severity increasing!)
- **Recovery monitoring** (severity decreasing)
- Answering: *"How severe is it RIGHT NOW?"*

---

## Implementation in Code

### DataFrame Columns

```python
from runtheory import calculate_timeseries

ts = calculate_timeseries(spi, threshold=-1.2)

# Available columns:
ts['magnitude_cumulative']      # Total deficit (debt analogy)
ts['magnitude_instantaneous']   # Current severity (NDVI analogy)
ts['deficit']                   # Same as magnitude_instantaneous
```

### Plotting Both

```python
from visualization import plot_event_timeline

# Default plot shows both magnitude types
fig = plot_event_timeline(ts)
# Creates 5 panels:
#   1. Index with event periods
#   2. Duration
#   3. Magnitude (Cumulative) - blue color
#   4. Magnitude (Instantaneous) - red color
#   5. Intensity
```

### Manual Plotting

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel 1: Cumulative
ax1.fill_between(ts.index, ts['magnitude_cumulative'], 0,
                 color='steelblue', alpha=0.5)
ax1.set_title('Cumulative Magnitude (Total Deficit)')
ax1.set_ylabel('Total Accumulated Deficit')

# Panel 2: Instantaneous
ax2.fill_between(ts.index, ts['magnitude_instantaneous'], 0,
                 color='coral', alpha=0.5)
ax2.set_title('Instantaneous Magnitude (Current Severity)')
ax2.set_ylabel('Current Monthly Severity')
ax2.set_xlabel('Time')

plt.tight_layout()
plt.show()
```

---

## Mathematical Relationship

### During Active Event

**Cumulative** = Sum of all **Instantaneous** values from event start

```
magnitude_cumulative[t] = Σ magnitude_instantaneous[i] for i = event_start to t
```

**Intensity** = Average of **Instantaneous** values

```
intensity[t] = magnitude_cumulative[t] / duration[t]
            = mean(magnitude_instantaneous[i]) for i = event_start to t
```

---

## Literature Context

### Standard Run Theory (Yevjevich 1967)

- Original run theory used **cumulative magnitude** only
- Defined as total deficit below threshold
- Used for event-based analysis (complete events)

### Modern Operational Monitoring

- **Instantaneous magnitude** better for real-time monitoring
- Shows current conditions, not just accumulated history
- More intuitive for decision-makers ("How bad is it NOW?")
- Aligns with how other indices work (e.g., NDVI phenology)

### This Implementation

We provide **BOTH** to support all use cases:

- ✅ Traditional run theory analysis (cumulative)
- ✅ Modern operational monitoring (instantaneous)
- ✅ Clear documentation of differences
- ✅ Appropriate visualizations for each

---

## Common Questions

### Q1: Which one is "correct"?

**Both are correct!** They measure different things:

- Cumulative = "How much total deficit?"
- Instantaneous = "How severe is it now?"

### Q2: Why doesn't cumulative decrease during recovery?

Because it's a **total** (like debt). The total debt doesn't decrease just because you're spending less — it only resets when the event ends (debt paid off).

### Q3: Why does instantaneous look like NDVI?

Because both measure **current state** that varies with conditions:

- NDVI: Current greenness (varies with crop growth stage)
- Instantaneous magnitude: Current severity (varies with SPI value)

### Q4: Which should I use in my analysis?

Depends on your question:

- "Total impact?" → Cumulative
- "Current conditions?" → Instantaneous
- "Not sure?" → Use both!

### Q5: Can cumulative be zero while an event continues?

No! If an event is ongoing, cumulative is always > 0 (and increasing). But instantaneous CAN be zero during an event if SPI is exactly at the threshold.

---

## Example Analysis

### Scenario: Monitoring 2023 Dry Event

```python
import xarray as xr
from runtheory import calculate_timeseries
from visualization import plot_event_timeline

# Load SPI
spi = xr.open_dataarray('spi_12.nc').isel(lat=50, lon=100)

# Calculate time series
ts = calculate_timeseries(spi, threshold=-1.2)

# Extract 2023
ts_2023 = ts.loc['2023']

# Question 1: What was the total water deficit in 2023?
total_deficit = ts_2023['magnitude_cumulative'].max()
print(f"Total cumulative deficit: {total_deficit:.2f}")

# Question 2: When was the event most severe?
worst_month = ts_2023['magnitude_instantaneous'].idxmax()
worst_severity = ts_2023['magnitude_instantaneous'].max()
print(f"Worst month: {worst_month}")
print(f"Peak severity: {worst_severity:.2f}")

# Question 3: Is the event currently worsening or easing?
recent = ts_2023['magnitude_instantaneous'].tail(3)
if recent.is_monotonic_decreasing:
    print("Event is EASING (severity decreasing)")
elif recent.is_monotonic_increasing:
    print("Event is WORSENING (severity increasing)")
else:
    print("Event severity is FLUCTUATING")

# Visualize both
fig = plot_event_timeline(ts_2023)
plt.savefig('event_2023_analysis.png')
```

---

## Summary

| Aspect | Cumulative | Instantaneous |
|--------|-----------|---------------|
| **Definition** | Sum of all deficits | Current month's deficit |
| **Behavior** | Always increases | Varies with SPI |
| **Pattern** | Staircase up | Rise-peak-fall |
| **Analogy** | Debt accumulation | Crop NDVI phenology |
| **Units** | Index units (cumulative) | Index units/month |
| **Resets** | When event ends | When event ends |
| **Use for** | Total impact, event comparison | Current monitoring, peak detection |
| **Answers** | "How much total deficit?" | "How severe now?" |

---

**Both are valuable tools in your climate extremes analysis toolkit. Use them together for comprehensive understanding!**

## See Also

- [Run Theory Guide](runtheory.md) - Dry/wet event identification and analysis
- [Visualization Guide](visualization.md) - Plotting options
- [SPI Guide](spi.md) - Precipitation-only index
- [SPEI Guide](spei.md) - Temperature-inclusive index
