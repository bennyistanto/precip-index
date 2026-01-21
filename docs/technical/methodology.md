# Scientific Methodology

## Overview

This document describes the scientific methods and mathematical foundations underlying the climate indices and climate extreme event analysis in this package.

---

## Part 1: Climate Indices (SPI & SPEI)

### 1.1 Standardized Precipitation Index (SPI)

**Developed by:** McKee, Doesken, and Kleist (1993)

**Purpose:** Quantify precipitation deficit and surplus relative to long-term climatology for monitoring both dry (drought) and wet (flood/excess) conditions

**Interpretation:**
- **Negative values:** Dry conditions (drought)
- **Positive values:** Wet conditions (flooding/excess precipitation)

**Mathematical Foundation:**

#### Step 1: Temporal Aggregation

For a given time scale `n` (months), calculate rolling sum:

```
P_i^n = Σ(j=i-n+1 to i) P_j
```

Where:
- `P_i^n` = accumulated precipitation for n-month period ending at month i
- `P_j` = monthly precipitation

#### Step 2: Distribution Fitting

Fit gamma distribution to aggregated precipitation using Maximum Likelihood Estimation:

**Gamma PDF:**
```
f(x) = (1 / (β^α * Γ(α))) * x^(α-1) * e^(-x/β)
```

Where:
- `α` = shape parameter
- `β` = scale parameter
- `Γ(α)` = gamma function

**Parameter Estimation:**
```
α = (1 / 4A) * (1 + sqrt(1 + 4A/3))
β = x̄ / α

Where:
A = ln(x̄) - (Σln(x)) / n
x̄ = mean precipitation
```

#### Step 3: Probability Calculation

Calculate cumulative probability:

```
G(x) = ∫(0 to x) f(t) dt
```

Account for zero precipitation:

```
H(x) = q + (1-q) * G(x)
```

Where:
- `q` = probability of zero precipitation
- `1-q` = probability of non-zero precipitation

#### Step 4: Standardization

Transform to standard normal distribution:

```
SPI = Φ^(-1)(H(x))
```

Where:
- `Φ^(-1)` = inverse standard normal cumulative distribution function

**Interpretation:**

| SPI Value | Category |
|-----------|----------|
| ≥ 2.0 | Extremely wet |
| 1.5 to 1.99 | Very wet |
| 1.0 to 1.49 | Moderately wet |
| -0.99 to 0.99 | Near normal |
| -1.0 to -1.49 | Moderately dry |
| -1.5 to -1.99 | Severely dry |
| ≤ -2.0 | Extremely dry |

**Time Scales:**
- **SPI-1:** Monthly, sensitive to short-term conditions
- **SPI-3:** Seasonal, reflects soil moisture
- **SPI-6:** Medium-term, agricultural water stress
- **SPI-12:** Annual, hydrological drought
- **SPI-24:** Long-term, reservoir/groundwater drought

### 1.2 Standardized Precipitation Evapotranspiration Index (SPEI)

**Developed by:** Vicente-Serrano, Beguería, and López-Moreno (2010)

**Purpose:** Extend SPI by including temperature effects via evapotranspiration for monitoring both dry and wet conditions with climate change sensitivity

**Interpretation:**
- **Negative values:** Dry conditions (drought with temperature effects)
- **Positive values:** Wet conditions (surplus with temperature effects)

**Mathematical Foundation:**

#### Step 1: Water Balance

Calculate monthly water balance:

```
D_i = P_i - PET_i
```

Where:
- `D_i` = climatic water balance for month i
- `P_i` = precipitation
- `PET_i` = potential evapotranspiration

#### Step 2: Temporal Aggregation

Same as SPI, but on water balance:

```
D_i^n = Σ(j=i-n+1 to i) D_j
```

#### Step 3-4: Distribution and Standardization

Same process as SPI:
1. Fit log-logistic or gamma distribution to `D^n`
2. Calculate cumulative probability
3. Transform to standard normal

**Key Difference from SPI:**

SPEI includes temperature effects through PET, making it sensitive to:
- Rising temperatures (increases PET → more negative D)
- Climate change impacts
- Agricultural water stress

**PET Methods Used:**

**Thornthwaite Method:**
```
PET = 16 * (10*T/I)^a

Where:
T = monthly mean temperature (°C)
I = annual heat index = Σ(T_i/5)^1.514
a = 0.49239 + 1.7912*10^-2*I - 7.71*10^-5*I^2 + 6.75*10^-7*I^3
```

**Hargreaves Method:**
```
PET = 0.0023 * R_a * (T_mean + 17.8) * sqrt(T_max - T_min)

Where:
R_a = extraterrestrial radiation (function of latitude and day of year)
T_mean = (T_max + T_min) / 2
```

### 1.3 Calibration Period

**Standard Practice:** 30-year climatological normal

**WMO Recommendation:** 1991-2020 (current standard period)

**Purpose:**
- Establish local probability distribution
- Ensure stationary reference period
- Allow spatial and temporal comparisons

**Implementation:**
```python
spi_12 = spi(precip, scale=12,
             calibration_start_year=1991,
             calibration_end_year=2020)
```

---

## Part 2: Climate Extreme Event Analysis (Run Theory)

### 2.1 Run Theory

**Developed by:** Yevjevich (1967)

**Purpose:** Identify and characterize climate extreme events based on continuous periods beyond a threshold. Works for both dry (drought) and wet (flood/excess) events.

**Mathematical Foundation:**

#### Run Definition

A **run** is a continuous sequence where the variable remains below (or above) a threshold level.

**Truncation level** (threshold): `x_0`

**Drought run (dry events):** Continuous period where `SPI < x_0` (negative threshold)
**Wet run (wet events):** Continuous period where `SPI > x_0` (positive threshold)

#### Event Identification

**For drought events (negative threshold):**
**Start:** First month where `SPI_i < x_0`
**End:** Last month where `SPI_i < x_0` before recovery
**Recovery:** `SPI_i ≥ x_0`

**For wet events (positive threshold):**
**Start:** First month where `SPI_i > x_0`
**End:** Last month where `SPI_i > x_0` before return to normal
**Return:** `SPI_i ≤ x_0`

**Filtering:** Events shorter than minimum duration excluded

### 2.2 Climate Extreme Event Characteristics

#### Duration (D)

```
D = t_end - t_start + 1
```

Number of consecutive months below threshold.

**Units:** months

#### Magnitude (M) - Cumulative

```
M = Σ(i=start to end) |x_0 - SPI_i|
```

Total accumulated deviation from threshold during event (deficit for drought, surplus for wet).

**Units:** Same as SPI (standardized units)

**Interpretation:**
- Represents total water deficit
- Monotonically increases during event
- Analogous to cumulative financial debt

#### Magnitude (M_inst) - Instantaneous

```
M_inst(t) = x_0 - SPI_t    for each month t in event
```

Current severity at specific time.

**Units:** Same as SPI

**Interpretation:**
- Varies during drought (rises-peaks-falls)
- Like NDVI crop phenology
- Shows drought evolution pattern

**Relationship:**
```
M = Σ M_inst(t)    for all t in event
```

#### Intensity (I)

```
I = M / D
```

Average severity per month.

**Units:** SPI units per month

**Interpretation:**
- High I: severe but possibly short drought
- Low I: mild but possibly long drought

#### Peak (P)

```
P = min(SPI_i)    for i in [start, end]
```

Most severe SPI value during event.

**Units:** SPI units

**Interpretation:**
- Worst moment of drought
- Related to maximum instantaneous magnitude
- `P = M_inst(t_peak)`

#### Peak Date

```
t_peak = argmin(SPI_i)    for i in [start, end]
```

When peak severity occurred.

#### Inter-arrival Time (T)

```
T_n = t_start(n+1) - t_start(n)
```

Time between consecutive drought onsets.

**Units:** months

**Interpretation:**
- Drought frequency
- Return period approximation

### 2.3 Period Aggregation

**Purpose:** Answer decision-maker questions about specific time periods

**Mathematical Foundation:**

For a spatial grid and time period [t_start, t_end]:

#### Number of Events

```
N(x,y) = count of events at location (x,y) during period
```

#### Total Drought Months

```
D_total(x,y) = Σ D_i    for all events i at (x,y)
```

#### Total Magnitude

```
M_total(x,y) = Σ M_i    for all events i at (x,y)
```

#### Mean Magnitude

```
M_mean(x,y) = M_total(x,y) / N(x,y)
```

#### Maximum Magnitude

```
M_max(x,y) = max(M_i)    for all events i at (x,y)
```

#### Worst Peak

```
P_worst(x,y) = min(P_i)    for all events i at (x,y)
```

#### Percent Time in Drought

```
Pct(x,y) = (D_total(x,y) / n_months) * 100

Where n_months = (t_end - t_start + 1) in months
```

**Implementation:**
- Applied independently to each grid cell
- Parallelizable across spatial domain
- Efficient for large gridded datasets

---

## Part 3: Statistical Considerations

### 3.1 Why Gamma Distribution?

**Original SPI (McKee et al., 1993):** Used gamma distribution

**Advantages:**
- Bounded at zero (like precipitation)
- Flexible shape (α parameter)
- Robust globally
- Well-established theory

**Alternative (Pearson Type III):**
- Sometimes used for SPI
- More flexible (3 parameters)
- **Problem:** Fails in arid regions
  - L-moments estimation unstable
  - Division by zero with low variability
  - Not recommended for global datasets

**This Package:** Gamma only (reliable, robust)

### 3.2 Zero Precipitation Handling

Gamma distribution defined for x > 0, but precipitation can be zero.

**Solution:**

```
H(x) = q + (1-q) * G(x)

Where:
q = P(precipitation = 0) = n_zeros / n_total
```

**Mixed distribution:**
- Discrete component at zero (probability q)
- Continuous component (gamma) for x > 0

### 3.3 Minimum Data Requirements

**For reliable SPI:**
- Minimum: 30 years of monthly data (360 months)
- Recommended: 50+ years for robust statistics
- **Critical:** Data must be quality-controlled

**For drought event analysis:**
- Minimum: 20 years (to identify events)
- Recommended: 30+ years (for return periods)

### 3.4 Spatial Consistency

**Important:** Each grid cell fitted independently

**Reason:**
- Different climate regimes
- Different precipitation distributions
- Spatial heterogeneity

**Result:** SPI=-1.0 means same probability (~16th percentile) everywhere

---

## Part 4: Operational Considerations

### 4.1 Near-Real-Time Updates

**Approach:**
1. Fit distribution on historical calibration period (1991-2020)
2. Save fitted parameters
3. For new data, use pre-fitted parameters

**Advantage:**
- Consistent with historical baseline
- Fast operational updates
- No refitting required

**Implementation:**
```python
# One-time calibration
params = spi(precip_calibration, scale=12, ...)
params.to_netcdf('spi_params.nc')

# Operational use
spi_new = spi(precip_new, scale=12, fitting_params=params)
```

### 4.2 Threshold Selection

**Common Thresholds:**

| Threshold | Percentile | Use Case |
|-----------|------------|----------|
| -0.5 | ~31st | Early warning |
| -0.8 | ~21st | Agricultural concern |
| -1.0 | ~16th | Standard operational |
| -1.2 | ~11st | Conservative threshold |
| -1.5 | ~7th | Severe drought |
| -2.0 | ~2nd | Extreme drought |

**Recommendation:** -1.0 or -1.2 for operational monitoring

**Rationale:**
- Balances sensitivity vs false alarms
- Captures significant droughts
- Allows minimum duration filtering

### 4.3 Minimum Duration

**Purpose:** Filter short-term fluctuations

**Typical Values:**
- SPI-1: min_duration = 2-3 months
- SPI-3: min_duration = 2-3 months (already smoothed)
- SPI-6: min_duration = 2 months
- SPI-12: min_duration = 3 months (captures sustained events)

**Impact:**
- Higher = fewer, longer events
- Lower = more events, includes brief dry spells

---

## Part 5: Validation and Quality Control

### 5.1 Input Data Quality

**Requirements:**
- CF-compliant NetCDF
- Monthly temporal resolution
- Regular spatial grid
- Minimal missing data (<5%)

**Checks:**
- No negative precipitation
- Reasonable value ranges (0-1000 mm/month)
- Temporal continuity
- Spatial consistency

### 5.2 Distribution Fitting Quality

**Checks:**
- Parameters within reasonable ranges
  - Shape (α): typically 0.5-5
  - Scale (β): positive, reasonable magnitude
- Fitted distribution matches data
- No extreme outliers

**Fallback:**
- If fitting fails: use default parameters
- Issue warning to user
- Set problematic cells to NaN

### 5.3 Output Validation

**Checks:**
- SPI/SPEI range typically -3 to +3
- Mean ≈ 0, StdDev ≈ 1 (over calibration period)
- No excessive NaN values
- Spatial patterns reasonable

---

## Part 6: Limitations and Assumptions

### 6.1 SPI Limitations

1. **Precipitation only:** Ignores temperature, evapotranspiration
2. **Stationarity:** Assumes constant climate (violated under climate change)
3. **Distribution choice:** Gamma may not fit all locations perfectly
4. **Data requirements:** Needs long, quality-controlled records

### 6.2 SPEI Limitations

1. **PET uncertainty:** Different methods give different results
2. **Data requirements:** Needs temperature + precipitation
3. **Stationarity:** Climate change affects both P and PET
4. **Complexity:** More inputs = more potential errors

### 6.3 Run Theory Limitations

1. **Threshold dependence:** Results sensitive to threshold choice
2. **Minimum duration:** Arbitrary choice affects event count
3. **Independence:** Adjacent events may not be truly independent
4. **Spatial correlation:** Events often span multiple grid cells (not captured in point analysis)

### 6.4 Assumptions

1. **Monthly data:** Sub-monthly processes not captured
2. **Point analysis:** Each grid cell independent
3. **Calibration period:** Representative of long-term climate
4. **Linear trend:** No explicit detrending (assumes stationary)

---

## Part 7: Best Practices

### 7.1 Index Selection

**Use SPI when:**
- Only precipitation data available
- Purely meteorological drought
- Comparison with global studies (most common)
- Simplicity desired

**Use SPEI when:**
- Temperature data available
- Agricultural drought (crop water stress)
- Climate change analysis
- Temperature effects important

### 7.2 Time Scale Selection

| Scale | Application |
|-------|-------------|
| SPI-1 | Month-to-month variability, not recommended for drought monitoring |
| SPI-3 | Seasonal drought, soil moisture |
| SPI-6 | Agricultural season, crop impacts |
| SPI-12 | Hydrological year, water resources |
| SPI-24 | Long-term drought, groundwater, reservoirs |

**Recommendation:** Use SPI-12 for general drought monitoring

### 7.3 Calibration Period

**Best Practice:**
- Use most recent 30-year WMO normal period (currently 1991-2020)
- Ensure data quality in calibration period
- Document clearly in metadata

**When to Update:**
- Every 10 years (WMO updates)
- After significant station changes
- For climate change studies (use fixed historical baseline)

### 7.4 Threshold and Duration

**Standard Approach:**
- Threshold: -1.0 or -1.2
- Minimum duration: 3 months for SPI-12

**Sensitivity Analysis:**
- Test multiple thresholds
- Evaluate against known events
- Document choice rationale

---

## References

### SPI
- McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales. 8th Conference on Applied Climatology, 17-22 January, Anaheim, CA.

- Guttman, N.B. (1999). Accepting the Standardized Precipitation Index: a calculation algorithm. Journal of the American Water Resources Association, 35(2), 311-322.

- WMO (2012). Standardized Precipitation Index User Guide (WMO-No. 1090). Geneva, Switzerland.

### SPEI
- Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. Journal of Climate, 23(7), 1696-1718.

- Beguería, S., Vicente-Serrano, S.M., Reig, F., Latorre, B. (2014). Standardized precipitation evapotranspiration index (SPEI) revisited: parameter fitting, evapotranspiration models, tools, datasets and drought monitoring. International Journal of Climatology, 34(10), 3001-3023.

### Run Theory
- Yevjevich, V. (1967). An objective approach to definitions and investigations of continental hydrologic droughts. Hydrology Papers 23, Colorado State University, Fort Collins, CO.

- Tallaksen, L.M., van Lanen, H.A.J. (2004). Hydrological Drought: Processes and Estimation Methods for Streamflow and Groundwater. Developments in Water Science 48, Elsevier, Amsterdam.

### PET
- Thornthwaite, C.W. (1948). An approach toward a rational classification of climate. Geographical Review, 38(1), 55-94.

- Hargreaves, G.H., Samani, Z.A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture, 1(2), 96-99.

---

## See Also

- [Implementation Details](implementation.md) - Code architecture
- [API Reference](api-reference.md) - Function documentation
- [User Guides](../user-guide/) - Practical usage
