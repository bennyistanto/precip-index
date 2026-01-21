"""
Climate extreme event identification using run theory.

Implements run theory for identifying and analyzing climate extreme events
(both dry/drought and wet/flood conditions) from SPI/SPEI time series,
including duration, magnitude, severity, and inter-arrival time.

Works with threshold direction:
- Negative thresholds (e.g., -1.2): Identify dry events (drought)
- Positive thresholds (e.g., +1.2): Identify wet events (flooding/excess)

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank

References:
    - Yevjevich, V. (1967). An objective approach to definitions and investigations
      of continental hydrologic droughts. Hydrology Papers, Colorado State University.
    - Shukla, S., & Wood, A.W. (2008). Use of a standardized runoff index for
      characterizing hydrologic drought. Geophysical Research Letters, 35(2).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr

from config import get_logger

# Module logger
_logger = get_logger(__name__)


# =============================================================================
# CORE RUN THEORY FUNCTIONS
# =============================================================================

def identify_runs(
    values: np.ndarray,
    threshold: float = -1.0,
    below: bool = True
) -> List[Dict[str, Union[int, float]]]:
    """
    Identify runs (consecutive periods) below or above a threshold.

    Run theory identifies periods where values continuously remain below
    (or above) a specified threshold level.

    :param values: 1-D array of index values (e.g., SPI, SPEI)
    :param threshold: threshold value for run identification (default: -1.0)
    :param below: if True, identify runs below threshold; if False, above threshold
    :return: list of dictionaries containing run characteristics

    Each dictionary contains:
        - 'start_idx': starting index of the run (0-based)
        - 'end_idx': ending index of the run (inclusive)
        - 'duration': length of the run in time steps
        - 'magnitude': cumulative sum of deviations from threshold
        - 'intensity': average deviation from threshold (magnitude/duration)
        - 'peak': minimum (or maximum) value during the run
        - 'peak_idx': index of peak value

    Example:
        >>> spi = np.array([0.5, -1.2, -1.5, -0.8, 0.3, -1.1, -1.3])
        >>> events = identify_runs(spi, threshold=-1.0)
        >>> len(events)
        2
        >>> events[0]['duration']
        3
    """
    # Validate input
    if len(values) == 0:
        return []

    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return []

    # Identify where condition is met
    if below:
        condition = values < threshold
    else:
        condition = values > threshold

    # Combine with valid mask
    condition = condition & valid_mask

    # Find run starts and ends
    # Pad with False to detect runs at boundaries
    padded = np.concatenate(([False], condition, [False]))
    diff = np.diff(padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    # Extract run characteristics
    runs = []
    for start, end in zip(starts, ends):
        run_values = values[start:end+1]
        duration = end - start + 1

        # Calculate magnitude (cumulative deviation from threshold)
        if below:
            deviations = threshold - run_values  # Positive values
            magnitude = np.sum(deviations)
            peak = np.min(run_values)
            peak_idx = start + np.argmin(run_values)
        else:
            deviations = run_values - threshold  # Positive values
            magnitude = np.sum(deviations)
            peak = np.max(run_values)
            peak_idx = start + np.argmax(run_values)

        intensity = magnitude / duration

        runs.append({
            'start_idx': int(start),
            'end_idx': int(end),
            'duration': int(duration),
            'magnitude': float(magnitude),
            'intensity': float(intensity),
            'peak': float(peak),
            'peak_idx': int(peak_idx)
        })

    return runs


def calculate_interarrival_times(runs: List[Dict]) -> List[int]:
    """
    Calculate inter-arrival times between consecutive climate extreme events.

    Inter-arrival time (T) is the duration from the start of one event
    to the start of the next event, including both event and non-event periods.

    :param runs: list of run dictionaries from identify_runs()
    :return: list of inter-arrival times (length = len(runs) - 1)

    Note:
        Inter-arrival time is useful for climatological analysis but may be
        ambiguous for continuous monitoring since different event patterns
        can produce the same inter-arrival value.

    Example:
        >>> runs = [{'start_idx': 10, 'end_idx': 15}, {'start_idx': 25, 'end_idx': 30}]
        >>> calculate_interarrival_times(runs)
        [15]  # 25 - 10 = 15 months
    """
    if len(runs) <= 1:
        return []

    interarrival = []
    for i in range(len(runs) - 1):
        t = runs[i+1]['start_idx'] - runs[i]['start_idx']
        interarrival.append(t)

    return interarrival


# =============================================================================
# EVENT-BASED CLIMATE EXTREME CHARACTERISTICS
# =============================================================================

def identify_events(
    index_values: Union[np.ndarray, xr.DataArray, pd.Series],
    threshold: float = -1.0,
    min_duration: int = 1
) -> pd.DataFrame:
    """
    Identify discrete climate extreme events and their characteristics.

    Works for both dry (drought) and wet (flood/excess) events based on
    threshold direction:
    - Negative threshold (e.g., -1.2): Identifies dry events (drought)
    - Positive threshold (e.g., +1.2): Identifies wet events (flooding)

    This function is suitable for historical analysis where you want to
    extract complete events from a time series.

    :param index_values: 1-D array of SPI/SPEI values
    :param threshold: event threshold (negative for dry, positive for wet, default: -1.0)
    :param min_duration: minimum duration to be considered an event (default: 1)
    :return: DataFrame with event characteristics

    DataFrame columns:
        - event_id: sequential event number (1, 2, 3, ...)
        - start_idx: starting index
        - end_idx: ending index
        - start_date: starting date (if index_values is time-aware)
        - end_date: ending date (if index_values is time-aware)
        - duration: event duration (months)
        - magnitude: cumulative deviation from threshold
        - intensity: average deviation (magnitude/duration)
        - peak: most extreme value during event
        - peak_idx: index of peak
        - peak_date: date of peak (if time-aware)
        - interarrival: time to next event (NaN for last event)

    Example:
        >>> spi = xr.open_dataarray('spi_12.nc').isel(lat=0, lon=0)
        >>> # Dry events (drought)
        >>> dry_events = identify_events(spi, threshold=-1.2)
        >>> print(f"Found {len(dry_events)} drought events")
        >>> # Wet events (flooding)
        >>> wet_events = identify_events(spi, threshold=+1.2)
        >>> print(f"Found {len(wet_events)} wet events")
    """
    # Convert to numpy array and extract time index if available
    time_index = None
    if isinstance(index_values, xr.DataArray):
        if 'time' in index_values.dims:
            time_index = pd.to_datetime(index_values.time.values)
        index_values = index_values.values
    elif isinstance(index_values, pd.Series):
        time_index = index_values.index
        index_values = index_values.values

    # Identify runs - determine direction based on threshold sign
    # Negative threshold (dry): look for values below threshold
    # Positive threshold (wet): look for values above threshold
    below = threshold < 0
    runs = identify_runs(index_values, threshold=threshold, below=below)

    # Filter by minimum duration
    runs = [r for r in runs if r['duration'] >= min_duration]

    if len(runs) == 0:
        _logger.warning("No events found with specified criteria")
        return pd.DataFrame()

    # Calculate inter-arrival times
    interarrival_times = calculate_interarrival_times(runs)
    interarrival_times.append(np.nan)  # Last event has no inter-arrival

    # Build DataFrame
    events = []
    for i, run in enumerate(runs, start=1):
        event = {
            'event_id': i,
            'start_idx': run['start_idx'],
            'end_idx': run['end_idx'],
            'duration': run['duration'],
            'magnitude': run['magnitude'],
            'intensity': run['intensity'],
            'peak': run['peak'],
            'peak_idx': run['peak_idx'],
            'interarrival': interarrival_times[i-1]
        }

        # Add dates if time index available
        if time_index is not None:
            event['start_date'] = time_index[run['start_idx']]
            event['end_date'] = time_index[run['end_idx']]
            event['peak_date'] = time_index[run['peak_idx']]

        events.append(event)

    df = pd.DataFrame(events)

    # Set appropriate column order
    if time_index is not None:
        cols = ['event_id', 'start_idx', 'end_idx', 'start_date', 'end_date',
                'duration', 'magnitude', 'intensity', 'peak', 'peak_idx',
                'peak_date', 'interarrival']
    else:
        cols = ['event_id', 'start_idx', 'end_idx', 'duration', 'magnitude',
                'intensity', 'peak', 'peak_idx', 'interarrival']

    return df[cols]


# =============================================================================
# TIME-SERIES CLIMATE EXTREME MONITORING
# =============================================================================

def calculate_timeseries(
    index_values: Union[np.ndarray, xr.DataArray, pd.Series],
    threshold: float = -1.0
) -> pd.DataFrame:
    """
    Calculate climate extreme event characteristics as time series for continuous monitoring.

    Works for both dry (drought) and wet (flood) events based on threshold direction:
    - Negative threshold (e.g., -1.2): Monitors dry events (drought)
    - Positive threshold (e.g., +1.2): Monitors wet events (flooding)

    Unlike identify_events(), this returns a time series where each
    time step shows the current event state and cumulative characteristics.

    This is useful for:
    - Real-time event monitoring
    - Visualizing how event characteristics evolve
    - Month-by-month tracking

    :param index_values: 1-D array of SPI/SPEI values
    :param threshold: event threshold (negative for dry, positive for wet, default: -1.0)
    :return: DataFrame with time series of event characteristics

    DataFrame columns:
        - time: time index (if available)
        - index_value: original SPI/SPEI value
        - is_event: boolean indicating if exceeding threshold
        - event_id: current event number (0 if no event)
        - duration: cumulative duration of current event (0 if no event)
        - magnitude_cumulative: total accumulated deviation (like debt, always increasing)
        - magnitude_instantaneous: current month's deviation (like NDVI, varies with index)
        - intensity: current intensity (magnitude_cumulative/duration, 0 if no event)
        - peak_so_far: most extreme value in current event (0 if no event)
        - deviation: current deviation from threshold (same as magnitude_instantaneous)

    Note on magnitude variables:
        - magnitude_cumulative: Sum of all monthly deviations within event. Always increases
          during event (represents total impact). Use for: total impact assessment,
          event comparison, magnitude statistics.

        - magnitude_instantaneous: Current month's deviation from threshold. Varies with
          SPI/SPEI (rises when event worsens, falls when event eases). Like crop NDVI
          phenology. Use for: monitoring event evolution, identifying peaks, real-time tracking.

    Example:
        >>> spi = xr.open_dataarray('spi_12.nc').isel(lat=0, lon=0)
        >>> # Monitor drought (dry) events
        >>> dry_ts = calculate_timeseries(spi, threshold=-1.2)
        >>> dry_ts[dry_ts.is_event].plot(x='time', y='magnitude_cumulative')
        >>> # Monitor wet events
        >>> wet_ts = calculate_timeseries(spi, threshold=+1.2)
        >>> wet_ts[wet_ts.is_event].plot(x='time', y='magnitude_cumulative')
    """
    # Convert to numpy array and extract time index if available
    time_index = None
    if isinstance(index_values, xr.DataArray):
        if 'time' in index_values.dims:
            time_index = pd.to_datetime(index_values.time.values)
        index_values = index_values.values
    elif isinstance(index_values, pd.Series):
        time_index = index_values.index
        index_values = index_values.values

    n = len(index_values)

    # Initialize arrays
    is_event = np.zeros(n, dtype=bool)
    event_id = np.zeros(n, dtype=int)
    duration = np.zeros(n, dtype=int)
    magnitude = np.zeros(n, dtype=float)
    intensity = np.zeros(n, dtype=float)
    peak_so_far = np.zeros(n, dtype=float)
    deviation = np.zeros(n, dtype=float)

    # Track current event
    current_event = 0
    current_duration = 0
    current_magnitude = 0.0
    current_peak = 0.0

    # Determine event logic based on threshold direction
    # Negative threshold (dry): below threshold, use abs() for deviation
    # Positive threshold (wet): above threshold, use abs() for deviation
    is_dry = threshold < 0

    for i in range(n):
        if np.isnan(index_values[i]):
            # Skip NaN values but reset current event
            if current_event > 0:
                current_event = 0
                current_duration = 0
                current_magnitude = 0.0
                current_peak = 0.0
            continue

        # Check if in event based on threshold direction
        in_event = (index_values[i] < threshold) if is_dry else (index_values[i] > threshold)

        if in_event:
            # In event
            is_event[i] = True

            # Start new event if needed
            if current_event == 0:
                current_event += 1
                current_duration = 1
                current_magnitude = abs(index_values[i] - threshold)
                current_peak = index_values[i]
            else:
                # Continue current event
                current_duration += 1
                current_magnitude += abs(index_values[i] - threshold)
                # Update peak: min for dry (more negative), max for wet (more positive)
                current_peak = min(current_peak, index_values[i]) if is_dry else max(current_peak, index_values[i])

            event_id[i] = current_event
            duration[i] = current_duration
            magnitude[i] = current_magnitude
            intensity[i] = current_magnitude / current_duration
            peak_so_far[i] = current_peak
            deviation[i] = abs(index_values[i] - threshold)
        else:
            # Not in event - reset current event
            if current_event > 0:
                current_event += 1  # Increment for next potential event
                current_duration = 0
                current_magnitude = 0.0
                current_peak = 0.0

    # Build DataFrame
    data = {
        'index_value': index_values,
        'is_event': is_event,
        'event_id': event_id,
        'duration': duration,
        'magnitude_cumulative': magnitude,
        'magnitude_instantaneous': deviation,
        'intensity': intensity,
        'peak_so_far': peak_so_far,
        'deviation': deviation
    }

    if time_index is not None:
        data['time'] = time_index
        df = pd.DataFrame(data)
        df = df.set_index('time')
    else:
        df = pd.DataFrame(data)

    return df


# =============================================================================
# SPATIAL CLIMATE EXTREME CHARACTERISTICS
# =============================================================================

def calculate_events_spatial(
    index_data: xr.DataArray,
    threshold: float = -1.0,
    min_duration: int = 1
) -> xr.Dataset:
    """
    Calculate climate extreme event characteristics for gridded (spatial) data.

    Works for both dry (drought) and wet (flood) events based on threshold direction.
    Applies event identification to each grid point and returns spatial statistics.

    :param index_data: xarray DataArray with dims (time, lat, lon) or (time, space)
    :param threshold: event threshold (negative for dry, positive for wet, default: -1.0)
    :param min_duration: minimum event duration (default: 1)
    :return: xarray Dataset with spatial event statistics

    Dataset variables:
        - num_events: total number of events at each location
        - mean_duration: average event duration
        - max_duration: maximum event duration
        - mean_magnitude: average event magnitude
        - max_magnitude: maximum event magnitude
        - mean_intensity: average event intensity
        - max_intensity: maximum event intensity
        - mean_peak: average peak severity
        - min_peak: most severe peak value

    Example:
        >>> spi = xr.open_dataarray('spi_12.nc')  # (time, lat, lon)
        >>> # Drought (dry) events
        >>> dry_stats = calculate_events_spatial(spi, threshold=-1.2)
        >>> dry_stats.num_events.plot()
        >>> # Wet events
        >>> wet_stats = calculate_events_spatial(spi, threshold=+1.2)
        >>> wet_stats.num_events.plot()
    """
    _logger.info(f"Calculating climate extreme event characteristics for {index_data.sizes} grid")

    # Ensure time is first dimension
    if 'time' not in index_data.dims:
        raise ValueError("DataArray must have 'time' dimension")

    # Stack spatial dimensions
    if set(index_data.dims) == {'time', 'lat', 'lon'}:
        stacked = index_data.stack(space=['lat', 'lon'])
    elif len(index_data.dims) == 2 and 'time' in index_data.dims:
        # Already has time + one spatial dimension
        spatial_dim = [d for d in index_data.dims if d != 'time'][0]
        stacked = index_data.rename({spatial_dim: 'space'})
    else:
        raise ValueError(f"Unsupported dimensions: {index_data.dims}")

    n_space = stacked.sizes['space']

    # Initialize output arrays
    num_events = np.zeros(n_space, dtype=int)
    mean_duration = np.full(n_space, np.nan, dtype=float)
    max_duration = np.full(n_space, np.nan, dtype=float)
    mean_magnitude = np.full(n_space, np.nan, dtype=float)
    max_magnitude = np.full(n_space, np.nan, dtype=float)
    mean_intensity = np.full(n_space, np.nan, dtype=float)
    max_intensity = np.full(n_space, np.nan, dtype=float)
    mean_peak = np.full(n_space, np.nan, dtype=float)
    min_peak = np.full(n_space, np.nan, dtype=float)

    # Process each location
    for i in range(n_space):
        if i % 1000 == 0:
            _logger.info(f"Processing location {i}/{n_space}")

        ts = stacked.isel(space=i).values

        # Identify events
        events_df = identify_events(ts, threshold=threshold, min_duration=min_duration)

        if len(events_df) > 0:
            num_events[i] = len(events_df)
            mean_duration[i] = events_df['duration'].mean()
            max_duration[i] = events_df['duration'].max()
            mean_magnitude[i] = events_df['magnitude'].mean()
            max_magnitude[i] = events_df['magnitude'].max()
            mean_intensity[i] = events_df['intensity'].mean()
            max_intensity[i] = events_df['intensity'].max()
            mean_peak[i] = events_df['peak'].mean()
            min_peak[i] = events_df['peak'].min()

    # Create output dataset
    ds = xr.Dataset(
        {
            'num_events': (['space'], num_events),
            'mean_duration': (['space'], mean_duration),
            'max_duration': (['space'], max_duration),
            'mean_magnitude': (['space'], mean_magnitude),
            'max_magnitude': (['space'], max_magnitude),
            'mean_intensity': (['space'], mean_intensity),
            'max_intensity': (['space'], max_intensity),
            'mean_peak': (['space'], mean_peak),
            'min_peak': (['space'], min_peak),
        },
        coords={'space': stacked.space}
    )

    # Unstack to original dimensions
    if 'lat' in index_data.dims and 'lon' in index_data.dims:
        ds = ds.unstack('space')

    # Add metadata
    ds.attrs['threshold'] = threshold
    ds.attrs['min_duration'] = min_duration
    ds.attrs['description'] = 'Climate extreme event characteristics from run theory'

    # Add variable attributes
    ds['num_events'].attrs = {
        'long_name': 'Number of climate extreme events',
        'units': 'count'
    }
    ds['mean_duration'].attrs = {
        'long_name': 'Mean event duration',
        'units': 'months'
    }
    ds['mean_magnitude'].attrs = {
        'long_name': 'Mean event magnitude',
        'units': 'index units',
        'description': 'Cumulative deviation from threshold'
    }
    ds['mean_intensity'].attrs = {
        'long_name': 'Mean event intensity',
        'units': 'index units',
        'description': 'Average deviation per month'
    }

    _logger.info("Spatial event characteristics calculation complete")

    return ds


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def summarize_events(events_df: pd.DataFrame) -> pd.Series:
    """
    Generate summary statistics of climate extreme events.

    Works for both dry (drought) and wet (flood) event summaries.

    :param events_df: DataFrame from identify_events()
    :return: Series with summary statistics

    Example:
        >>> # Drought events
        >>> dry_events = identify_events(spi, threshold=-1.2)
        >>> dry_summary = summarize_events(dry_events)
        >>> print(dry_summary)
        >>> # Wet events
        >>> wet_events = identify_events(spi, threshold=+1.2)
        >>> wet_summary = summarize_events(wet_events)
    """
    if len(events_df) == 0:
        return pd.Series({'num_events': 0})

    summary = pd.Series({
        'num_events': len(events_df),
        'mean_duration': events_df['duration'].mean(),
        'median_duration': events_df['duration'].median(),
        'max_duration': events_df['duration'].max(),
        'total_event_months': events_df['duration'].sum(),
        'mean_magnitude': events_df['magnitude'].mean(),
        'max_magnitude': events_df['magnitude'].max(),
        'mean_intensity': events_df['intensity'].mean(),
        'max_intensity': events_df['intensity'].max(),
        'mean_peak': events_df['peak'].mean(),
        'most_severe_peak': events_df['peak'].min(),
        'mean_interarrival': events_df['interarrival'].mean() if 'interarrival' in events_df else np.nan,
    })

    return summary


def get_event_state(
    index_value: float,
    threshold: float = -1.0
) -> Tuple[bool, str, float]:
    """
    Get current climate extreme event state from index value.

    Works for both dry (drought) and wet (flood) thresholds.

    :param index_value: current SPI/SPEI value
    :param threshold: event threshold (negative for dry, positive for wet)
    :return: tuple of (is_event, category, deviation)

    Example:
        >>> # Drought (dry) event
        >>> is_event, category, deviation = get_event_state(-1.5, threshold=-1.0)
        >>> print(f"Event: {is_event}, Category: {category}, Deviation: {deviation:.2f}")
        >>> # Wet event
        >>> is_event, category, deviation = get_event_state(+1.5, threshold=+1.0)
        >>> print(f"Event: {is_event}, Category: {category}, Deviation: {deviation:.2f}")
    """
    if np.isnan(index_value):
        return False, "No Data", 0.0

    # Determine if in event based on threshold direction
    if threshold < 0:
        # Dry events (drought) - below threshold
        is_event = index_value < threshold
        deviation = max(0.0, threshold - index_value)

        # Categorize severity (McKee et al., 1993)
        if index_value >= threshold:
            category = "No Event"
        elif index_value >= -1.5:
            category = "Moderate Drought"
        elif index_value >= -2.0:
            category = "Severe Drought"
        else:
            category = "Extreme Drought"
    else:
        # Wet events - above threshold
        is_event = index_value > threshold
        deviation = max(0.0, index_value - threshold)

        # Categorize severity
        if index_value <= threshold:
            category = "No Event"
        elif index_value <= 1.5:
            category = "Moderately Wet"
        elif index_value <= 2.0:
            category = "Very Wet"
        else:
            category = "Extremely Wet"

    return is_event, category, deviation


# =============================================================================
# TEMPORAL AGGREGATION FOR DECISION MAKERS
# =============================================================================

def calculate_period_statistics(
    index_data: xr.DataArray,
    threshold: float = -1.0,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    min_duration: int = 1
) -> xr.Dataset:
    """
    Calculate climate extreme event statistics for a specific time period (gridded output).

    Works for both dry (drought) and wet (flood) events based on threshold direction.

    This function answers decision-maker questions like:
    - "How many dry/wet events occurred in 2023?"
    - "What was the total event magnitude during 2021-2025?"
    - "Which areas had the worst events in the last 5 years?"

    :param index_data: xarray DataArray with dims (time, lat, lon) or (time, space)
    :param threshold: event threshold (negative for dry, positive for wet, default: -1.0)
    :param start_year: starting year for analysis (None = use all data)
    :param end_year: ending year for analysis (None = use all data)
    :param min_duration: minimum event duration to count (default: 1)
    :return: xarray Dataset with spatial statistics for the period

    Dataset variables:
        - num_events: number of events in period
        - total_event_months: total months in events
        - total_magnitude: sum of all event magnitudes
        - mean_magnitude: average magnitude per event
        - max_magnitude: largest single event magnitude
        - worst_peak: most extreme SPI/SPEI value
        - mean_intensity: average intensity across all events
        - max_intensity: maximum intensity from any event
        - pct_time_in_event: percentage of time in events

    Example:
        >>> spi = xr.open_dataarray('spi_12.nc')
        >>> # Drought events in 2023
        >>> dry_stats_2023 = calculate_period_statistics(spi, threshold=-1.2,
        ...                                              start_year=2023, end_year=2023)
        >>> # Wet events in last 5 years
        >>> wet_stats = calculate_period_statistics(spi, threshold=+1.2,
        ...                                         start_year=2020, end_year=2024)
        >>> # Map results
        >>> dry_stats_2023.num_events.plot()
    """
    _logger.info(f"Calculating period statistics for {start_year}-{end_year}")

    # Ensure time is first dimension
    if 'time' not in index_data.dims:
        raise ValueError("DataArray must have 'time' dimension")

    # Filter by time period if specified
    if start_year is not None or end_year is not None:
        time_values = pd.to_datetime(index_data.time.values)

        if start_year is not None:
            index_data = index_data.sel(time=time_values.year >= start_year)
        if end_year is not None:
            time_values = pd.to_datetime(index_data.time.values)
            index_data = index_data.sel(time=time_values.year <= end_year)

    # Stack spatial dimensions
    if set(index_data.dims) == {'time', 'lat', 'lon'}:
        stacked = index_data.stack(space=['lat', 'lon'])
    elif len(index_data.dims) == 2 and 'time' in index_data.dims:
        spatial_dim = [d for d in index_data.dims if d != 'time'][0]
        stacked = index_data.rename({spatial_dim: 'space'})
    else:
        raise ValueError(f"Unsupported dimensions: {index_data.dims}")

    n_space = stacked.sizes['space']
    n_time = stacked.sizes['time']

    # Initialize output arrays
    num_events = np.zeros(n_space, dtype=int)
    total_event_months = np.zeros(n_space, dtype=int)
    total_magnitude = np.full(n_space, np.nan, dtype=float)
    mean_magnitude = np.full(n_space, np.nan, dtype=float)
    max_magnitude = np.full(n_space, np.nan, dtype=float)
    worst_peak = np.full(n_space, np.nan, dtype=float)
    mean_intensity = np.full(n_space, np.nan, dtype=float)
    max_intensity = np.full(n_space, np.nan, dtype=float)
    pct_time_in_event = np.zeros(n_space, dtype=float)

    # Process each location
    for i in range(n_space):
        if i % 1000 == 0:
            _logger.info(f"Processing location {i}/{n_space}")

        ts = stacked.isel(space=i).values

        # Identify events
        events_df = identify_events(ts, threshold=threshold, min_duration=min_duration)

        if len(events_df) > 0:
            num_events[i] = len(events_df)
            total_event_months[i] = events_df['duration'].sum()
            total_magnitude[i] = events_df['magnitude'].sum()
            mean_magnitude[i] = events_df['magnitude'].mean()
            max_magnitude[i] = events_df['magnitude'].max()
            worst_peak[i] = events_df['peak'].min()
            mean_intensity[i] = events_df['intensity'].mean()
            max_intensity[i] = events_df['intensity'].max()
            pct_time_in_event[i] = (total_event_months[i] / n_time) * 100
        else:
            # No events - set to 0 instead of NaN
            total_magnitude[i] = 0.0
            pct_time_in_event[i] = 0.0

    # Create output dataset
    ds = xr.Dataset(
        {
            'num_events': (['space'], num_events),
            'total_event_months': (['space'], total_event_months),
            'total_magnitude': (['space'], total_magnitude),
            'mean_magnitude': (['space'], mean_magnitude),
            'max_magnitude': (['space'], max_magnitude),
            'worst_peak': (['space'], worst_peak),
            'mean_intensity': (['space'], mean_intensity),
            'max_intensity': (['space'], max_intensity),
            'pct_time_in_event': (['space'], pct_time_in_event),
        },
        coords={'space': stacked.space}
    )

    # Unstack to original dimensions
    if 'lat' in index_data.dims and 'lon' in index_data.dims:
        ds = ds.unstack('space')

    # Add metadata
    ds.attrs['threshold'] = threshold
    ds.attrs['min_duration'] = min_duration
    ds.attrs['start_year'] = start_year if start_year else 'all'
    ds.attrs['end_year'] = end_year if end_year else 'all'
    ds.attrs['description'] = f'Climate extreme event statistics for {start_year}-{end_year}'

    # Add variable attributes
    ds['num_events'].attrs = {
        'long_name': 'Number of climate extreme events',
        'units': 'count',
        'description': f'Total events during {start_year}-{end_year}'
    }
    ds['total_event_months'].attrs = {
        'long_name': 'Total months in events',
        'units': 'months',
        'description': 'Cumulative time in event condition'
    }
    ds['total_magnitude'].attrs = {
        'long_name': 'Total event magnitude',
        'units': 'index units',
        'description': 'Sum of all event magnitudes in period'
    }
    ds['worst_peak'].attrs = {
        'long_name': 'Worst event peak',
        'units': 'index units',
        'description': 'Most extreme SPI/SPEI value in period'
    }
    ds['pct_time_in_event'].attrs = {
        'long_name': 'Percentage of time in events',
        'units': '%',
        'description': 'Proportion of period in event condition'
    }

    _logger.info("Period statistics calculation complete")

    return ds


def calculate_annual_statistics(
    index_data: xr.DataArray,
    threshold: float = -1.0,
    min_duration: int = 1
) -> xr.Dataset:
    """
    Calculate climate extreme event statistics for each year (gridded output).

    Works for both dry (drought) and wet (flood) events based on threshold direction.

    Answers questions like:
    - "Which year had the most events?"
    - "What was the trend in event magnitude over time?"
    - "Compare 2015 vs 2023 event severity"

    :param index_data: xarray DataArray with dims (time, lat, lon)
    :param threshold: event threshold (negative for dry, positive for wet, default: -1.0)
    :param min_duration: minimum event duration (default: 1)
    :return: xarray Dataset with dimensions (year, lat, lon)

    Dataset variables (same as calculate_period_statistics):
        - num_events: events per year
        - total_magnitude: total magnitude per year
        - worst_peak: worst value per year
        - pct_time_in_event: % of year in events
        - etc.

    Example:
        >>> spi = xr.open_dataarray('spi_12.nc')
        >>> # Annual drought statistics
        >>> dry_annual = calculate_annual_statistics(spi, threshold=-1.2)
        >>> # Compare specific years
        >>> stats_2015 = dry_annual.sel(year=2015)
        >>> stats_2023 = dry_annual.sel(year=2023)
        >>> # Plot time series
        >>> dry_annual.num_events.mean(dim=['lat', 'lon']).plot()
    """
    _logger.info("Calculating annual event statistics")

    # Get unique years
    time_values = pd.to_datetime(index_data.time.values)
    years = np.unique(time_values.year)

    # Calculate stats for each year
    annual_datasets = []
    for year in years:
        _logger.info(f"Processing year {year}")
        ds = calculate_period_statistics(
            index_data,
            threshold=threshold,
            start_year=int(year),
            end_year=int(year),
            min_duration=min_duration
        )
        annual_datasets.append(ds)

    # Combine into single dataset with year dimension
    combined = xr.concat(annual_datasets, dim='year')
    combined['year'] = years

    # Update attributes
    combined.attrs['description'] = 'Annual climate extreme event statistics'
    combined.attrs['threshold'] = threshold

    _logger.info("Annual event statistics calculation complete")

    return combined


def compare_periods(
    index_data: xr.DataArray,
    periods: List[Tuple[int, int]],
    threshold: float = -1.0,
    min_duration: int = 1,
    period_names: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Compare climate extreme event statistics across multiple time periods.

    Works for both dry (drought) and wet (flood) events based on threshold direction.

    Answers questions like:
    - "Compare last 5 years (2020-2024) vs previous 5 years (2015-2019)"
    - "How does 2023 compare to the historical average (1991-2020)?"
    - "Compare El Niño years vs La Niña years"

    :param index_data: xarray DataArray with dims (time, lat, lon)
    :param periods: list of (start_year, end_year) tuples
    :param threshold: event threshold (negative for dry, positive for wet, default: -1.0)
    :param min_duration: minimum event duration (default: 1)
    :param period_names: optional names for each period
    :return: xarray Dataset with 'period' dimension

    Example:
        >>> spi = xr.open_dataarray('spi_12.nc')
        >>>
        >>> # Compare drought events: recent vs historical
        >>> periods = [(1991, 2020), (2021, 2024)]
        >>> names = ['Historical', 'Recent']
        >>> comparison = compare_periods(spi, periods, threshold=-1.2, period_names=names)
        >>>
        >>> # Calculate difference
        >>> diff = comparison.sel(period='Recent') - comparison.sel(period='Historical')
        >>> diff.num_events.plot(title='Change in Events')
    """
    _logger.info(f"Comparing {len(periods)} periods")

    # Generate default names if not provided
    if period_names is None:
        period_names = [f"{start}-{end}" for start, end in periods]

    if len(period_names) != len(periods):
        raise ValueError("Number of period names must match number of periods")

    # Calculate stats for each period
    period_datasets = []
    for (start, end), name in zip(periods, period_names):
        _logger.info(f"Processing period: {name} ({start}-{end})")
        ds = calculate_period_statistics(
            index_data,
            threshold=threshold,
            start_year=start,
            end_year=end,
            min_duration=min_duration
        )
        period_datasets.append(ds)

    # Combine into single dataset with period dimension
    combined = xr.concat(period_datasets, dim='period')
    combined['period'] = period_names

    # Update attributes
    combined.attrs['description'] = 'Comparison of climate extreme event statistics across periods'
    combined.attrs['threshold'] = threshold
    combined.attrs['periods'] = str(periods)

    _logger.info("Period comparison complete")

    return combined
