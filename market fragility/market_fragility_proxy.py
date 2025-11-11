import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =============================================================================

# Window sizes (in number of bars, e.g., minutes for 1-min data)
FAST_WINDOW = 60      # 1 hour for short-term volatility and impact
SLOW_WINDOW = 240     # 4 hours for vol-of-vol and kurtosis
Z_WINDOW = 240        # 4 hours for Z-score normalization

# Safety constant to prevent division by zero
EPSILON = 1e-10

# Plotting style configuration
plt.style.use('seaborn-v0_8-darkgrid')


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================

def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate 1-period log returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Close' prices
        
    Returns
    -------
    pd.Series
        Log returns series
    """
    return np.log(df['Close'] / df['Close'].shift(1))


def calculate_volatility_of_volatility(returns: pd.Series, 
                                       fast_window: int, 
                                       slow_window: int) -> pd.Series:
    """
    Proxy 1: Volatility of Volatility (Market Instability)
    
    High vol-of-vol indicates that volatility itself is erratic and unpredictable,
    suggesting a fragile market structure.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    fast_window : int
        Window for calculating short-term volatility (e.g., 60 for 1 hour)
    slow_window : int
        Window for calculating volatility of the short-term volatility
        
    Returns
    -------
    pd.Series
        Volatility of volatility proxy
    """
    # Step 1: Calculate fast volatility (rolling std of returns)
    fast_vol = returns.rolling(window=fast_window, min_periods=fast_window//2).std()
    
    # Step 2: Calculate the volatility OF volatility (std of fast vol)
    vol_of_vol = fast_vol.rolling(window=slow_window, min_periods=slow_window//2).std()
    
    return vol_of_vol


def calculate_price_impact(df: pd.DataFrame, 
                           fast_window: int) -> pd.Series:
    """
    Proxy 2: Price Impact / Illiquidity (Market Thinness)
    
    Measures how much price moves per unit of volume. High values indicate
    that small volumes cause large price swings - a sign of illiquidity and
    fragility.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'High', 'Low', and 'Volume' columns
    fast_window : int
        Window for calculating average impact
        
    Returns
    -------
    pd.Series
        Price impact proxy (illiquidity measure)
    """
    # Calculate instantaneous impact: price range per unit volume
    instantaneous_impact = (df['High'] - df['Low']) / (df['Volume'] + EPSILON)
    
    # Smooth it over the fast window to reduce noise
    impact_proxy = instantaneous_impact.rolling(
        window=fast_window, 
        min_periods=fast_window//2
    ).mean()
    
    return impact_proxy


def calculate_tail_risk(returns: pd.Series, 
                        slow_window: int) -> pd.Series:
    """
    Proxy 3: Tail Risk / Kurtosis (Chaos Metric)
    
    High kurtosis (> 3) indicates fat tails - extreme events are becoming
    more likely than a normal distribution would suggest. This signals
    increasing market fragility and potential for sudden jumps.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    slow_window : int
        Window for calculating rolling kurtosis
        
    Returns
    -------
    pd.Series
        Rolling kurtosis (tail risk proxy)
    """
    # Calculate rolling kurtosis
    # Using pandas' rolling().apply() for kurtosis calculation
    from scipy.stats import kurtosis
    
    tail_risk = returns.rolling(
        window=slow_window, 
        min_periods=slow_window//2
    ).apply(lambda x: kurtosis(x, fisher=False, nan_policy='omit'), raw=True)
    
    return tail_risk


def normalize_proxy(proxy: pd.Series, 
                    z_window: int) -> pd.Series:
    """
    Normalize a proxy using rolling Z-score normalization.
    
    Z-score = (value - rolling_mean) / rolling_std
    
    This makes different proxies comparable by expressing them in terms
    of standard deviations from their recent mean.
    
    Parameters
    ----------
    proxy : pd.Series
        Raw proxy values to normalize
    z_window : int
        Window size for calculating rolling mean and std
        
    Returns
    -------
    pd.Series
        Z-score normalized proxy
    """
    rolling_mean = proxy.rolling(window=z_window, min_periods=z_window//2).mean()
    rolling_std = proxy.rolling(window=z_window, min_periods=z_window//2).std()
    
    # Prevent division by zero
    rolling_std = rolling_std.replace(0, EPSILON)
    
    z_score = (proxy - rolling_mean) / rolling_std
    
    return z_score


# =============================================================================
# MAIN FRAGILITY CALCULATION FUNCTION
# =============================================================================

def calculate_fragility(df: pd.DataFrame,
                       fast_window: int = FAST_WINDOW,
                       slow_window: int = SLOW_WINDOW,
                       z_window: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate the Market Fragility Index from OHLCV data.
    
    This is the main function that orchestrates the entire calculation pipeline:
    1. Calculate three fragility proxies (vol-of-vol, impact, tail risk)
    2. Normalize each proxy using rolling Z-scores
    3. Aggregate into a single Fragility Index
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        Index should be datetime for proper time-series handling
    fast_window : int, optional
        Window for short-term calculations (default: 60 bars)
    slow_window : int, optional
        Window for longer-term calculations (default: 240 bars)
    z_window : int, optional
        Window for Z-score normalization (default: same as slow_window)
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - Returns: 1-period log returns
        - VolOfVol: Volatility of volatility proxy
        - Impact: Price impact / illiquidity proxy
        - TailRisk: Kurtosis / tail risk proxy
        - Z_VolOfVol: Normalized vol-of-vol
        - Z_Impact: Normalized impact
        - Z_TailRisk: Normalized tail risk
        - Fragility_Index: Final aggregated fragility index
        
    Examples
    --------
    >>> df = pd.read_csv('market_data.csv', parse_dates=['Date'], index_col='Date')
    >>> df_with_fragility = calculate_fragility(df)
    >>> print(df_with_fragility[['Close', 'Fragility_Index']].tail())
    """
    # Use slow_window for z_window if not specified
    if z_window is None:
        z_window = slow_window
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("=" * 70)
    print("MARKET FRAGILITY PROXY - CALCULATION PIPELINE")
    print("=" * 70)
    print(f"Input data shape: {result_df.shape}")
    print(f"Date range: {result_df.index[0]} to {result_df.index[-1]}")
    print(f"Fast window: {fast_window} bars")
    print(f"Slow window: {slow_window} bars")
    print(f"Z-score window: {z_window} bars")
    print()
    
    # Step 1: Calculate returns
    print("Step 1/7: Calculating log returns...")
    result_df['Returns'] = calculate_returns(result_df)
    
    # Step 2: Calculate Proxy 1 - Volatility of Volatility
    print("Step 2/7: Calculating Volatility of Volatility (instability proxy)...")
    result_df['VolOfVol'] = calculate_volatility_of_volatility(
        result_df['Returns'], 
        fast_window, 
        slow_window
    )
    
    # Step 3: Calculate Proxy 2 - Price Impact / Illiquidity
    print("Step 3/7: Calculating Price Impact (illiquidity proxy)...")
    result_df['Impact'] = calculate_price_impact(result_df, fast_window)
    
    # Step 4: Calculate Proxy 3 - Tail Risk / Kurtosis
    print("Step 4/7: Calculating Tail Risk / Kurtosis (chaos proxy)...")
    result_df['TailRisk'] = calculate_tail_risk(result_df['Returns'], slow_window)
    
    # Step 5: Normalize all proxies
    print("Step 5/7: Normalizing proxies (rolling Z-scores)...")
    result_df['Z_VolOfVol'] = normalize_proxy(result_df['VolOfVol'], z_window)
    result_df['Z_Impact'] = normalize_proxy(result_df['Impact'], z_window)
    result_df['Z_TailRisk'] = normalize_proxy(result_df['TailRisk'], z_window)
    
    # Step 6: Aggregate into final Fragility Index
    print("Step 6/7: Aggregating into Fragility Index...")
    result_df['Fragility_Index'] = (
        result_df['Z_VolOfVol'] + 
        result_df['Z_Impact'] + 
        result_df['Z_TailRisk']
    ) / 3
    
    # Step 7: Summary statistics
    print("Step 7/7: Calculation complete!")
    print()
    print("=" * 70)
    print("FRAGILITY INDEX STATISTICS")
    print("=" * 70)
    
    # Only calculate stats for non-NaN values
    valid_fragility = result_df['Fragility_Index'].dropna()
    if len(valid_fragility) > 0:
        print(f"Mean Fragility Index: {valid_fragility.mean():.4f}")
        print(f"Std Fragility Index: {valid_fragility.std():.4f}")
        print(f"Min Fragility Index: {valid_fragility.min():.4f}")
        print(f"Max Fragility Index: {valid_fragility.max():.4f}")
        print(f"Current Fragility Index: {valid_fragility.iloc[-1]:.4f}")
        print()
        
        # Flag extreme fragility
        if valid_fragility.iloc[-1] > 2.0:
            print("⚠️  WARNING: EXTREME FRAGILITY DETECTED (Index > 2.0)")
        elif valid_fragility.iloc[-1] > 1.0:
            print("⚠️  CAUTION: ELEVATED FRAGILITY (Index > 1.0)")
        else:
            print("✓  Market fragility within normal range")
    else:
        print("Insufficient data for statistics calculation")
    
    print("=" * 70)
    print()
    
    return result_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_fragility_index(df: pd.DataFrame, 
                         figsize: Tuple[int, int] = (14, 8),
                         save_path: Optional[str] = "outputs/") -> None:
    """
    Plot 1: Main Chart - Price and Fragility Index
    
    Creates a 2-panel chart:
    - Top panel: Close price with color-coded background (fragility level)
    - Bottom panel: Fragility Index with threshold lines
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Close' and 'Fragility_Index' columns
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the plot to this path
    """
    # Filter to valid data
    plot_df = df[['Close', 'Fragility_Index']].dropna()
    
    if len(plot_df) == 0:
        print("Error: No valid data to plot")
        return
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle('Market Fragility Analysis', fontsize=16, fontweight='bold')
    
    # -----------------------------
    # Panel 1: Close Price
    # -----------------------------
    ax1 = axes[0]
    
    # Plot the close price
    ax1.plot(plot_df.index, plot_df['Close'], 
             color='#2E86AB', linewidth=1.5, label='Close Price')
    
    # Add background shading based on fragility level
    fragility = plot_df['Fragility_Index'].values
    
    # Define fragility zones
    low_fragility = fragility < 0
    moderate_fragility = (fragility >= 0) & (fragility < 1.0)
    high_fragility = (fragility >= 1.0) & (fragility < 2.0)
    extreme_fragility = fragility >= 2.0
    
    # Shade background (very light colors)
    ax1.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                     where=low_fragility, alpha=0.1, color='green', 
                     label='Low Fragility', interpolate=True)
    ax1.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                     where=moderate_fragility, alpha=0.1, color='yellow',
                     label='Moderate Fragility', interpolate=True)
    ax1.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                     where=high_fragility, alpha=0.1, color='orange',
                     label='High Fragility', interpolate=True)
    ax1.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                     where=extreme_fragility, alpha=0.15, color='red',
                     label='Extreme Fragility', interpolate=True)
    
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title('Asset Price with Fragility Regime Shading', fontsize=13)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # -----------------------------
    # Panel 2: Fragility Index
    # -----------------------------
    ax2 = axes[1]
    
    # Plot the fragility index
    ax2.plot(plot_df.index, plot_df['Fragility_Index'], 
             color='#A23B72', linewidth=2, label='Fragility Index')
    
    # Add horizontal threshold lines
    ax2.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Normal (0)')
    ax2.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Elevated (1.0)')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Extreme (2.0)')
    
    # Shade regions
    ax2.fill_between(plot_df.index, -10, 0, alpha=0.1, color='green')
    ax2.fill_between(plot_df.index, 0, 1.0, alpha=0.1, color='yellow')
    ax2.fill_between(plot_df.index, 1.0, 2.0, alpha=0.1, color='orange')
    ax2.fill_between(plot_df.index, 2.0, 10, alpha=0.1, color='red')
    
    # Set y-limits for better visualization
    y_min = max(plot_df['Fragility_Index'].min() - 0.5, -3)
    y_max = min(plot_df['Fragility_Index'].max() + 0.5, 5)
    ax2.set_ylim(y_min, y_max)
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fragility Index (Z-score)', fontsize=12, fontweight='bold')
    ax2.set_title('Market Fragility Index (Combined Metric)', fontsize=13)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path+'Fragility_index', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_fragility_components(df: pd.DataFrame,
                              figsize: Tuple[int, int] = (14, 10),
                              save_path: Optional[str] = "outputs/") -> None:
    """
    Plot 2: Component Dashboard - Individual Normalized Proxies
    
    Creates a 3-panel chart showing each normalized proxy separately:
    - Panel 1: Z_VolOfVol (Volatility of Volatility)
    - Panel 2: Z_Impact (Price Impact / Illiquidity)
    - Panel 3: Z_TailRisk (Kurtosis / Tail Risk)
    
    This helps understand which component is driving the final fragility index.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with normalized proxy columns
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the plot to this path
    """
    # Check for required columns
    required_cols = ['Z_VolOfVol', 'Z_Impact', 'Z_TailRisk']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Filter to valid data
    plot_df = df[required_cols].dropna()
    
    if len(plot_df) == 0:
        print("Error: No valid data to plot")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle('Fragility Component Dashboard (Normalized Proxies)', 
                 fontsize=16, fontweight='bold')
    
    # Component definitions
    components = [
        {
            'col': 'Z_VolOfVol',
            'title': 'Volatility of Volatility (Market Instability)',
            'color': '#E63946',
            'description': 'High values indicate erratic, unpredictable volatility'
        },
        {
            'col': 'Z_Impact',
            'title': 'Price Impact / Illiquidity (Market Thinness)',
            'color': '#F77F00',
            'description': 'High values indicate low liquidity, large price swings'
        },
        {
            'col': 'Z_TailRisk',
            'title': 'Tail Risk / Kurtosis (Chaos Metric)',
            'color': '#06A77D',
            'description': 'High values indicate fat tails, extreme events likely'
        }
    ]
    
    # Plot each component
    for idx, (ax, comp) in enumerate(zip(axes, components)):
        # Plot the component
        ax.plot(plot_df.index, plot_df[comp['col']], 
                color=comp['color'], linewidth=1.5, label=comp['col'])
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(y=2.0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(y=-1.0, color='blue', linestyle='--', linewidth=0.8, alpha=0.3)
        
        # Shade extreme regions
        ax.fill_between(plot_df.index, 2.0, 10, alpha=0.1, color='red')
        ax.fill_between(plot_df.index, -10, -2.0, alpha=0.1, color='blue')
        
        # Set y-limits
        y_min = max(plot_df[comp['col']].min() - 0.5, -4)
        y_max = min(plot_df[comp['col']].max() + 0.5, 5)
        ax.set_ylim(y_min, y_max)
        
        # Labels and styling
        ax.set_ylabel('Z-score', fontsize=11, fontweight='bold')
        ax.set_title(f"{comp['title']}\n({comp['description']})", 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        
        # Add current value annotation
        current_val = plot_df[comp['col']].iloc[-1]
        ax.annotate(f'Current: {current_val:.2f}', 
                   xy=(plot_df.index[-1], current_val),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                   fontsize=9, fontweight='bold')
    
    # Set x-label only on bottom plot
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path+"Fragility_component", dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def generate_fragility_report(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive fragility analysis report.
    
    Returns a dictionary with key metrics and interpretations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with fragility calculations
        
    Returns
    -------
    dict
        Report containing statistics and interpretations
    """
    report = {}
    
    # Current fragility metrics
    if 'Fragility_Index' in df.columns:
        fragility = df['Fragility_Index'].dropna()
        if len(fragility) > 0:
            current = fragility.iloc[-1]
            report['current_fragility'] = current
            report['mean_fragility'] = fragility.mean()
            report['std_fragility'] = fragility.std()
            report['percentile_rank'] = (fragility < current).sum() / len(fragility) * 100
            
            # Interpretation
            if current > 2.0:
                report['risk_level'] = 'EXTREME'
                report['interpretation'] = 'Market showing severe fragility. High risk of crashes or extreme volatility.'
            elif current > 1.0:
                report['risk_level'] = 'HIGH'
                report['interpretation'] = 'Market fragility elevated. Increased caution warranted.'
            elif current > 0:
                report['risk_level'] = 'MODERATE'
                report['interpretation'] = 'Market fragility above average but manageable.'
            else:
                report['risk_level'] = 'LOW'
                report['interpretation'] = 'Market fragility below average. Relatively stable conditions.'
    
    # Component contributions
    if all(col in df.columns for col in ['Z_VolOfVol', 'Z_Impact', 'Z_TailRisk']):
        components = df[['Z_VolOfVol', 'Z_Impact', 'Z_TailRisk']].dropna()
        if len(components) > 0:
            latest = components.iloc[-1]
            report['component_values'] = {
                'vol_of_vol': latest['Z_VolOfVol'],
                'impact': latest['Z_Impact'],
                'tail_risk': latest['Z_TailRisk']
            }
            
            # Identify dominant driver
            abs_vals = latest.abs()
            dominant = abs_vals.idxmax()
            report['dominant_driver'] = dominant.replace('Z_', '')
            report['dominant_value'] = latest[dominant]
    
    return report


# =============================================================================
# LIVE DATA ADAPTATION
# =============================================================================

def calculate_fragility_live(historical_buffer: pd.DataFrame,
                             new_data: pd.DataFrame,
                             buffer_size: int = 1000,
                             fast_window: int = FAST_WINDOW,
                             slow_window: int = SLOW_WINDOW) -> Tuple[pd.DataFrame, dict]:
    """
    Calculate fragility for live/streaming data.
    
    This function maintains a rolling buffer of recent data and calculates
    fragility on the combined historical + new data, then returns only the
    new rows with their fragility scores.
    
    USAGE FOR LIVE TRADING:
    -----------------------
    1. Initialize with historical data (e.g., last 1000 bars)
    2. As new bars arrive, call this function with the new bar(s)
    3. The function returns the fragility score for the new bar(s)
    4. Update your historical buffer with the returned DataFrame
    
    Parameters
    ----------
    historical_buffer : pd.DataFrame
        Rolling buffer of recent historical OHLCV data
    new_data : pd.DataFrame
        New incoming OHLCV bar(s) to process
    buffer_size : int
        Maximum size of the rolling buffer (older data is dropped)
    fast_window : int
        Fast window parameter
    slow_window : int
        Slow window parameter
        
    Returns
    -------
    tuple
        (updated_buffer, fragility_metrics)
        - updated_buffer: New rolling buffer to use for next iteration
        - fragility_metrics: Dict with current fragility values
        
    Examples
    --------
    >>> # Initialize with historical data
    >>> buffer = pd.read_csv('historical_data.csv', parse_dates=['Date'], index_col='Date')
    >>> buffer = buffer.tail(1000)  # Keep last 1000 bars
    >>> 
    >>> # In your live data loop:
    >>> while True:
    >>>     new_bar = fetch_latest_bar()  # Your data feed function
    >>>     buffer, metrics = calculate_fragility_live(buffer, new_bar)
    >>>     
    >>>     print(f"Current Fragility: {metrics['fragility_index']:.2f}")
    >>>     
    >>>     if metrics['fragility_index'] > 2.0:
    >>>         send_alert("EXTREME FRAGILITY DETECTED!")
    >>>     
    >>>     time.sleep(60)  # Wait for next bar
    """
    # Combine historical buffer with new data
    combined_data = pd.concat([historical_buffer, new_data])
    
    # Remove duplicates (in case of overlap)
    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
    
    # Sort by index (time)
    combined_data = combined_data.sort_index()
    
    # Keep only the most recent buffer_size bars
    if len(combined_data) > buffer_size:
        combined_data = combined_data.tail(buffer_size)
    
    # Calculate fragility on the full buffer
    combined_with_fragility = calculate_fragility(
        combined_data,
        fast_window=fast_window,
        slow_window=slow_window,
        z_window=slow_window
    )
    
    # Extract metrics for the new data only
    new_indices = new_data.index
    new_with_fragility = combined_with_fragility.loc[new_indices]
    
    # Prepare metrics dictionary
    if 'Fragility_Index' in new_with_fragility.columns:
        latest = new_with_fragility.iloc[-1]
        metrics = {
            'timestamp': latest.name,
            'fragility_index': latest.get('Fragility_Index', np.nan),
            'vol_of_vol': latest.get('Z_VolOfVol', np.nan),
            'impact': latest.get('Z_Impact', np.nan),
            'tail_risk': latest.get('Z_TailRisk', np.nan),
            'close': latest.get('Close', np.nan)
        }
    else:
        metrics = {'error': 'Insufficient data for fragility calculation'}
    
    return combined_with_fragility, metrics


# =============================================================================
# USAGE EXAMPLE AND TEST CODE
# =============================================================================

if __name__ == "__main__":
    """
    USAGE EXAMPLE: Batch Historical Analysis
    
    This section demonstrates how to:
    1. Load historical OHLCV data from a CSV
    2. Calculate the fragility index
    3. Generate visualizations
    4. Create a summary report
    """
    
    print("\n" + "="*70)
    print("MARKET FRAGILITY PROXY - USAGE EXAMPLE")
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # OPTION 1: Load Real Data from CSV
    # -------------------------------------------------------------------------
    
    try:
        # Try to load the Nifty 50 data from the workspace
        print("Attempting to load Nifty 50 data from workspace...")
        df = pd.read_csv('/Users/hemantsoni/Documents/AlgoFinance/NIFTY 50_minute.csv')
        
        # Parse the date column (adjust column name as needed)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
            df.set_index(date_columns[0], inplace=True)
        
        # Standardize column names to match OHLCV format
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower or 'vol' in col_lower:
                column_mapping[col] = 'Volume'
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Keep only the last 5000 bars for faster processing
        df = df.tail(5000)
        
        print(f"✓ Successfully loaded real data: {len(df)} bars")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Generating synthetic demo data instead...")
        
        # -------------------------------------------------------------------------
        # OPTION 2: Generate Synthetic Demo Data
        # -------------------------------------------------------------------------
        
        # Create synthetic data with realistic patterns
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=2000, freq='1min')
        
        # Generate price with trending + noise + occasional jumps
        price = 20000
        prices = [price]
        volumes = []
        
        for i in range(1, len(dates)):
            # Add trend, mean reversion, and noise
            drift = np.random.normal(0, 5)
            jump = np.random.normal(0, 50) if np.random.random() < 0.01 else 0
            price = price + drift + jump
            prices.append(price)
            
            # Volume with some correlation to price movement
            volume = max(100000, np.random.lognormal(13, 0.5))
            volumes.append(volume)
        
        # Create OHLC from prices
        df = pd.DataFrame({
            'Close': prices,
            'Volume': [volumes[0]] + volumes
        }, index=dates)
        
        df['High'] = df['Close'] * (1 + np.random.uniform(0, 0.002, len(df)))
        df['Low'] = df['Close'] * (1 - np.random.uniform(0, 0.002, len(df)))
        df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
        
        print(f"✓ Generated synthetic demo data: {len(df)} bars")
    
    # -------------------------------------------------------------------------
    # Calculate Fragility Index
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("CALCULATING FRAGILITY INDEX...")
    print("-"*70 + "\n")
    
    df_with_fragility = calculate_fragility(
        df,
        fast_window=60,      # 1 hour for minute data
        slow_window=240,     # 4 hours for minute data
        z_window=240         # 4 hours for normalization
    )
    
    # -------------------------------------------------------------------------
    # Generate Fragility Report
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("FRAGILITY ANALYSIS REPORT")
    print("-"*70 + "\n")
    
    report = generate_fragility_report(df_with_fragility)
    
    print(f"Current Fragility Index: {report.get('current_fragility', 'N/A'):.4f}")
    print(f"Risk Level: {report.get('risk_level', 'N/A')}")
    print(f"Interpretation: {report.get('interpretation', 'N/A')}")
    print(f"Percentile Rank: {report.get('percentile_rank', 'N/A'):.1f}%")
    
    if 'component_values' in report:
        print("\nComponent Contributions:")
        print(f"  - Volatility of Volatility: {report['component_values']['vol_of_vol']:.4f}")
        print(f"  - Price Impact: {report['component_values']['impact']:.4f}")
        print(f"  - Tail Risk: {report['component_values']['tail_risk']:.4f}")
        print(f"\nDominant Driver: {report.get('dominant_driver', 'N/A')}")
    
    # -------------------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS...")
    print("-"*70 + "\n")
    
    # Plot 1: Main Chart (Price + Fragility Index)
    print("Creating Plot 1: Price and Fragility Index...")
    plot_fragility_index(df_with_fragility)
    
    # Plot 2: Component Dashboard
    print("Creating Plot 2: Fragility Components Dashboard...")
    plot_fragility_components(df_with_fragility)
    
    # -------------------------------------------------------------------------
    # Demonstrate Live Data Usage
    # -------------------------------------------------------------------------
    
    print("\n" + "-"*70)
    print("LIVE DATA EXAMPLE (Simulated)")
    print("-"*70 + "\n")
    
    print("Simulating live data feed with 5 new bars...")
    
    # Use the last 1000 bars as initial buffer
    initial_buffer = df_with_fragility.tail(1000)[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Simulate 5 new incoming bars
    last_close = df['Close'].iloc[-1]
    new_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=1), 
                              periods=5, freq='1min')
    
    for i, date in enumerate(new_dates):
        # Simulate new bar
        new_close = last_close + np.random.normal(0, 10)
        new_bar = pd.DataFrame({
            'Open': [last_close],
            'High': [new_close * 1.001],
            'Low': [new_close * 0.999],
            'Close': [new_close],
            'Volume': [np.random.lognormal(13, 0.5)]
        }, index=[date])
        
        # Calculate fragility for new bar
        initial_buffer, metrics = calculate_fragility_live(
            initial_buffer, 
            new_bar,
            buffer_size=1000
        )
        
        print(f"Bar {i+1} ({date}): Fragility = {metrics.get('fragility_index', 'N/A'):.4f}, "
              f"Close = {metrics.get('close', 'N/A'):.2f}")
        
        last_close = new_close
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Integrate this module into your trading system")
    print("2. Use calculate_fragility() for batch historical analysis")
    print("3. Use calculate_fragility_live() for real-time monitoring")
    print("4. Set up alerts based on fragility thresholds")
    print("5. Adjust window parameters based on your timeframe")
    print("\nFor daily data: use fast_window=20, slow_window=60")
    print("For minute data: use fast_window=60, slow_window=240")
    print("For tick data: use fast_window=1000, slow_window=5000")
    print("="*70 + "\n")
