"""
Visualization functions for climate indices and extreme event characteristics.

Provides plotting utilities for SPI/SPEI time series and climate extreme events (both dry and wet).

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams, 
with substantial modifications for multi-distribution support, 
bidirectional event analysis, and scalable processing.
---
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
from datetime import datetime

from utils import get_logger

# Module logger
_logger = get_logger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_location_filename(
    base_name: str,
    data: Union[xr.DataArray, pd.Series, pd.DataFrame],
    extension: str = 'png',
    output_dir: str = 'output/plots/single'
) -> str:
    """
    Generate filename with location coordinates for single-location outputs.

    :param base_name: Base name for file (e.g., 'plot_events')
    :param data: DataArray or Series containing coordinate information
    :param extension: File extension (default: 'png')
    :param output_dir: Output directory (default: 'output/plots/single')
    :return: Full path with location-based filename

    Example:
        >>> filename = generate_location_filename('plot_events', spi_ts)
        'output/plots/single/plot_events_lat31.82_lon-7.07.png'
    """
    import os

    # Extract coordinates
    lat, lon = None, None

    if isinstance(data, xr.DataArray):
        if 'lat' in data.coords:
            lat = float(data.lat.values)
        if 'lon' in data.coords:
            lon = float(data.lon.values)
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        # Check for lat/lon attributes
        if hasattr(data, 'lat'):
            lat = float(data.lat)
        if hasattr(data, 'lon'):
            lon = float(data.lon)

    # Build filename
    if lat is not None and lon is not None:
        filename = f"{base_name}_lat{lat:.2f}_lon{lon:.2f}.{extension}"
    else:
        filename = f"{base_name}.{extension}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, filename)


# =============================================================================
# DROUGHT INDEX TIME SERIES PLOTS
# =============================================================================

def plot_index(
    index_values: Union[np.ndarray, xr.DataArray, pd.Series],
    threshold: float = -1.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
    colors: Optional[dict] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot climate index time series with WMO color classification.

    :param index_values: 1-D array of SPI/SPEI values
    :param threshold: event threshold for reference line (default: -1.0)
    :param title: plot title (optional)
    :param figsize: figure size (width, height) in inches
    :param colors: custom color scheme dict with WMO categories
    :param ax: existing axes to plot on (optional)
    :return: matplotlib Axes object

    Example:
        >>> spi = xr.open_dataarray('spi_12.nc').isel(lat=0, lon=0)
        >>> ax = plot_index(spi, threshold=-1.2, title='SPI-12 Morocco')
        >>> plt.show()
    """
    # Extract time index and values
    time_index = None
    if isinstance(index_values, xr.DataArray):
        if 'time' in index_values.dims:
            time_index = pd.to_datetime(index_values.time.values)
        values = index_values.values
    elif isinstance(index_values, pd.Series):
        time_index = index_values.index
        values = index_values.values
    else:
        values = index_values

    if time_index is None:
        time_index = np.arange(len(values))

    # Official SPI/SPEI color scheme (WMO standard classification)
    if colors is None:
        colors = {
            'exceptionally_dry': '#760005',    # -2.00 and below
            'extremely_dry': '#ec0013',        # -2.00 to -1.50
            'severely_dry': '#ffa938',         # -1.50 to -1.20
            'moderately_dry': '#fdd28a',       # -1.20 to -0.70
            'abnormally_dry': '#fefe53',       # -0.70 to -0.50
            'near_normal': '#ffffff',          # -0.50 to +0.50
            'abnormally_moist': '#a2fd6e',     # +0.50 to +0.70
            'moderately_moist': '#00b44a',     # +0.70 to +1.20
            'very_moist': '#008180',           # +1.20 to +1.50
            'extremely_moist': '#2a23eb',      # +1.50 to +2.00
            'exceptionally_moist': '#a21fec'   # +2.00 and above
        }

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot bar chart with colors based on categories
    for i, (t, v) in enumerate(zip(time_index, values)):
        if np.isnan(v):
            continue

        # Determine color based on WMO SPI/SPEI classification
        if v <= -2.0:
            color = colors['exceptionally_dry']
        elif v <= -1.5:
            color = colors['extremely_dry']
        elif v <= -1.2:
            color = colors['severely_dry']
        elif v <= -0.7:
            color = colors['moderately_dry']
        elif v <= -0.5:
            color = colors['abnormally_dry']
        elif v <= 0.5:
            color = colors['near_normal']
        elif v <= 0.7:
            color = colors['abnormally_moist']
        elif v <= 1.2:
            color = colors['moderately_moist']
        elif v <= 1.5:
            color = colors['very_moist']
        elif v <= 2.0:
            color = colors['extremely_moist']
        else:
            color = colors['exceptionally_moist']

        ax.bar(t, v, width=20.0, color=color, edgecolor='none', alpha=0.9)

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold})')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Formatting
    ax.set_ylabel('Index Value', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Format x-axis for dates if applicable
    if isinstance(time_index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))

    # Add legend with WMO classification
    legend_elements = [
        mpatches.Patch(color=colors['exceptionally_dry'], label='Exceptionally Dry (≤ -2.0)'),
        mpatches.Patch(color=colors['extremely_dry'], label='Extremely Dry (-2.0 to -1.5)'),
        mpatches.Patch(color=colors['severely_dry'], label='Severely Dry (-1.5 to -1.2)'),
        mpatches.Patch(color=colors['moderately_dry'], label='Moderately Dry (-1.2 to -0.7)'),
        mpatches.Patch(color=colors['abnormally_dry'], label='Abnormally Dry (-0.7 to -0.5)'),
        mpatches.Patch(color=colors['near_normal'], label='Near Normal (-0.5 to +0.5)'),
        mpatches.Patch(color=colors['abnormally_moist'], label='Abnormally Moist (+0.5 to +0.7)'),
        mpatches.Patch(color=colors['moderately_moist'], label='Moderately Moist (+0.7 to +1.2)'),
        mpatches.Patch(color=colors['very_moist'], label='Very Moist (+1.2 to +1.5)'),
        mpatches.Patch(color=colors['extremely_moist'], label='Extremely Moist (+1.5 to +2.0)'),
        mpatches.Patch(color=colors['exceptionally_moist'], label='Exceptionally Moist (≥ +2.0)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9)

    plt.tight_layout()

    return ax


# =============================================================================
# DROUGHT EVENT PLOTS
# =============================================================================

def plot_events(
    index_values: Union[np.ndarray, xr.DataArray, pd.Series],
    events_df: pd.DataFrame,
    threshold: float = -1.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot climate index with highlighted events.

    :param index_values: 1-D array of SPI/SPEI values
    :param events_df: DataFrame from identify_events()
    :param threshold: event threshold
    :param title: plot title
    :param figsize: figure size
    :param ax: existing axes (optional)
    :return: matplotlib Axes object

    Example:
        >>> from runtheory import identify_events
        >>> spi = xr.open_dataarray('spi_12.nc').isel(lat=0, lon=0)
        >>> events = identify_events(spi, threshold=-1.2)
        >>> ax = plot_events(spi, events, threshold=-1.2)
        >>> plt.show()
    """
    # Extract time index and values
    time_index = None
    if isinstance(index_values, xr.DataArray):
        if 'time' in index_values.dims:
            time_index = pd.to_datetime(index_values.time.values)
        values = index_values.values
    elif isinstance(index_values, pd.Series):
        time_index = index_values.index
        values = index_values.values
    else:
        values = index_values

    if time_index is None:
        time_index = np.arange(len(values))

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot index as line
    ax.plot(time_index, values, color='black', linewidth=1.5, alpha=0.7, label='Index')

    # Determine event type based on threshold direction
    is_dry = threshold < 0

    # Choose color palette and marker based on event type
    if is_dry:
        event_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(events_df)))
        peak_color = 'darkred'
        peak_marker = 'v'  # Down arrow for dry
    else:
        event_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(events_df)))
        peak_color = 'darkblue'
        peak_marker = '^'  # Up arrow for wet

    for i, (idx, event) in enumerate(events_df.iterrows()):
        start = int(event['start_idx'])
        end = int(event['end_idx']) + 1

        event_time = time_index[start:end]
        event_vals = values[start:end]

        # Fill area between values and threshold
        # Dry events: fill below threshold (values < threshold)
        # Wet events: fill above threshold (values > threshold)
        if is_dry:
            where_condition = event_vals < threshold
        else:
            where_condition = event_vals > threshold

        ax.fill_between(event_time, event_vals, threshold,
                        where=where_condition,
                        color=event_colors[i], alpha=0.5,
                        label=f"Event {event['event_id']}" if i < 5 else None)

        # Mark peak
        peak_idx = int(event['peak_idx'])
        ax.scatter(time_index[peak_idx], values[peak_idx],
                  color=peak_color, s=50, zorder=5, marker=peak_marker)

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold})')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # Formatting
    ax.set_ylabel('Index Value', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Format x-axis for dates
    if isinstance(time_index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))

    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()

    return ax


def plot_event_characteristics(
    events_df: pd.DataFrame,
    characteristic: str = 'duration',
    figsize: Tuple[float, float] = (12, 5)
) -> plt.Figure:
    """
    Create multi-panel plot showing climate event characteristics.

    :param events_df: DataFrame from identify_events()
    :param characteristic: which characteristic to highlight ('duration', 'magnitude', 'intensity')
    :param figsize: figure size
    :return: matplotlib Figure object

    Creates a 2-panel figure:
        - Left: Bar chart of characteristic values per event
        - Right: Scatter plot showing relationship with other characteristics

    Example:
        >>> events = identify_events(spi, threshold=-1.2)
        >>> fig = plot_event_characteristics(events, characteristic='magnitude')
        >>> plt.show()
    """
    if len(events_df) == 0:
        _logger.warning("No events to plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Bar chart of selected characteristic
    x = events_df['event_id']
    y = events_df[characteristic]

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(events_df)))
    ax1.bar(x, y, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Event ID', fontsize=11)
    ax1.set_ylabel(f'{characteristic.capitalize()}', fontsize=11)
    ax1.set_title(f'Climate Event {characteristic.capitalize()}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right panel: Scatter plot relationships
    if characteristic == 'duration':
        ax2.scatter(events_df['duration'], events_df['magnitude'],
                   s=100, c=events_df['intensity'], cmap='YlOrRd',
                   edgecolor='black', linewidth=0.5, alpha=0.8)
        ax2.set_xlabel('Duration (months)', fontsize=11)
        ax2.set_ylabel('Magnitude', fontsize=11)
        ax2.set_title('Duration vs Magnitude (colored by Intensity)', fontsize=12)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Intensity', fontsize=10)

    elif characteristic == 'magnitude':
        ax2.scatter(events_df['magnitude'], events_df['peak'],
                   s=events_df['duration']*20, c=events_df['intensity'],
                   cmap='YlOrRd', edgecolor='black', linewidth=0.5, alpha=0.8)
        ax2.set_xlabel('Magnitude', fontsize=11)
        ax2.set_ylabel('Peak Severity', fontsize=11)
        ax2.set_title('Magnitude vs Peak (size=Duration, color=Intensity)', fontsize=12)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Intensity', fontsize=10)

    elif characteristic == 'intensity':
        ax2.scatter(events_df['intensity'], events_df['peak'],
                   s=events_df['duration']*20, c=events_df['magnitude'],
                   cmap='YlOrRd', edgecolor='black', linewidth=0.5, alpha=0.8)
        ax2.set_xlabel('Intensity', fontsize=11)
        ax2.set_ylabel('Peak Severity', fontsize=11)
        ax2.set_title('Intensity vs Peak (size=Duration, color=Magnitude)', fontsize=12)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Magnitude', fontsize=10)

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_event_timeline(
    timeseries_df: pd.DataFrame,
    characteristics: list = None,
    figsize: Tuple[float, float] = (14, 10)
) -> plt.Figure:
    """
    Plot time series of climate event characteristics for continuous monitoring.

    :param timeseries_df: DataFrame from calculate_timeseries()
    :param characteristics: list of characteristics to plot. If None, uses default set
                          including both magnitude types
    :param figsize: figure size
    :return: matplotlib Figure object

    Creates multi-panel plot showing evolution of event characteristics over time.
    By default, plots:
        - Index value with event periods
        - Duration
        - Magnitude (cumulative) - total accumulated deviation
        - Magnitude (instantaneous) - current severity (NDVI-like pattern)
        - Intensity

    Example:
        >>> from runtheory import calculate_timeseries
        >>> ts = calculate_timeseries(spi, threshold=-1.2)
        >>> fig = plot_event_timeline(ts)
        >>> plt.show()
    """
    # Default characteristics if not specified
    if characteristics is None:
        characteristics = ['duration', 'magnitude_cumulative',
                         'magnitude_instantaneous', 'intensity']
    n_panels = len(characteristics) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

    if n_panels == 1:
        axes = [axes]

    # Panel 1: Index values with event periods shaded
    ax = axes[0]
    event_mask = timeseries_df['is_event']

    ax.plot(timeseries_df.index, timeseries_df['index_value'],
            color='black', linewidth=1.5, label='Index')
    ax.fill_between(timeseries_df.index,
                    timeseries_df['index_value'].where(event_mask),
                    0, color='red', alpha=0.3, label='Event')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Index Value', fontsize=11)
    ax.set_title('Climate Index and Event Characteristics Timeline', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Remaining panels: Characteristics
    for i, char in enumerate(characteristics, start=1):
        ax = axes[i]

        # Only plot where event is active
        event_data = timeseries_df[char].where(event_mask)

        # Choose colors based on variable type
        if char == 'magnitude_cumulative':
            fill_color = 'steelblue'
            line_color = 'darkblue'
            label = 'Magnitude (Cumulative)\nTotal Accumulated Deviation'
        elif char == 'magnitude_instantaneous':
            fill_color = 'coral'
            line_color = 'darkred'
            label = 'Magnitude (Instantaneous)\nCurrent Monthly Severity'
        else:
            fill_color = 'orange'
            line_color = 'darkred'
            label = char.replace('_', ' ').title()

        ax.fill_between(timeseries_df.index, event_data, 0,
                       color=fill_color, alpha=0.5)
        ax.plot(timeseries_df.index, event_data,
               color=line_color, linewidth=1.5)

        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=11)

    # Format x-axis
    if isinstance(timeseries_df.index, pd.DatetimeIndex):
        axes[-1].xaxis.set_major_formatter(DateFormatter('%Y'))

    plt.tight_layout()
    return fig


# =============================================================================
# SPATIAL PLOTS
# =============================================================================

def plot_spatial_stats(
    event_stats: xr.Dataset,
    variable: str = 'num_events',
    cmap: str = 'YlOrRd',
    figsize: Tuple[float, float] = (12, 8),
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot spatial map of climate event statistics.

    :param event_stats: Dataset from calculate_events_spatial()
    :param variable: which variable to plot ('num_events', 'mean_duration', etc.)
    :param cmap: colormap name
    :param figsize: figure size
    :param title: plot title (optional)
    :return: matplotlib Axes object

    Example:
        >>> from runtheory import calculate_events_spatial
        >>> spi = xr.open_dataarray('spi_12.nc')
        >>> stats = calculate_events_spatial(spi, threshold=-1.2)
        >>> ax = plot_spatial_stats(stats, variable='mean_duration')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': None})

    # Plot
    im = event_stats[variable].plot(ax=ax, cmap=cmap, add_colorbar=True,
                                      cbar_kwargs={'label': event_stats[variable].attrs.get('long_name', variable)})

    if title is None:
        title = f"Spatial Distribution: {event_stats[variable].attrs.get('long_name', variable)}"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)

    plt.tight_layout()
    return ax
