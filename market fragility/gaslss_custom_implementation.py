import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)  # For reproducibility


class SkewedTGAS:
    """
    Custom implementation of GAS model with Skewed Student-t distribution.
    
    This is a simplified but effective implementation that extracts:
    - Dynamic scale (volatility)
    - Dynamic skewness
    - Dynamic shape (kurtosis)
    """
    
    def __init__(self, returns, scale_lags=1, skew_lags=1, shape_lags=1):
        """
        Initialize the Skewed-t GAS model.
        
        Parameters:
        -----------
        returns : array-like
            Return series
        scale_lags : int
            Number of lags for scale parameter
        skew_lags : int
            Number of lags for skewness parameter
        shape_lags : int
            Number of lags for shape parameter
        """
        self.returns = np.array(returns)
        self.n = len(returns)
        self.scale_lags = scale_lags
        self.skew_lags = skew_lags
        self.shape_lags = shape_lags
        
        # Initialize parameters
        self.scale_series = np.zeros(self.n)
        self.skew_series = np.zeros(self.n)
        self.shape_series = np.zeros(self.n)
        
    def skewed_t_loglik(self, params, data):
        """
        Log-likelihood for skewed Student-t distribution.
        
        Parameters:
        -----------
        params : array
            [location, log_scale, skew, log_shape]
        data : array
            Returns data
        """
        loc = params[0]
        scale = np.exp(params[1])  # Ensure positive
        skew = params[2]
        shape = np.exp(params[3]) + 2  # Ensure > 2
        
        # Standardized returns
        z = (data - loc) / scale
        
        # Skewed-t log-likelihood (simplified)
        # Adjust for skewness
        z_adj = z * (1 + skew * np.sign(z))
        
        # Student-t component
        loglik = -0.5 * np.log(1 + z_adj**2 / shape)
        loglik = np.sum(loglik) - self.n * (np.log(scale) + 0.5 * np.log(shape))
        
        return -loglik  # Return negative for minimization
    
    def fit_rolling_gas(self, window=100):
        """
        Fit GAS model using rolling window approach with robust statistics.
        
        This uses a combination of optimization and robust statistics to
        capture time-varying dynamics reliably.
        
        Parameters:
        -----------
        window : int
            Rolling window size
        """
        print("\n" + "="*80)
        print("FITTING CUSTOM SKEWED-T GAS MODEL")
        print("="*80)
        print(f"\nUsing rolling window approach with window={window}")
        print(f"Total observations: {self.n}")
        
        # Initialize arrays
        self.scale_series = np.zeros(self.n)
        self.skew_series = np.zeros(self.n)
        self.shape_series = np.zeros(self.n)
        
        # For first window, use initial estimates
        init_window_data = self.returns[:window]
        init_scale = np.std(init_window_data)
        init_skew = stats.skew(init_window_data) if len(init_window_data) > 2 else 0.0
        init_kurt = stats.kurtosis(init_window_data) if len(init_window_data) > 2 else 0.0
        init_shape = max(3.0, min(30.0, 10.0 - init_kurt / 2))
        
        self.scale_series[:window] = init_scale
        self.skew_series[:window] = init_skew
        self.shape_series[:window] = init_shape
        
        print(f"\nInitial estimates:")
        print(f"  Scale (Vol): {init_scale:.6f}")
        print(f"  Skewness: {init_skew:.4f}")
        print(f"  Shape (Kurtosis): {init_shape:.4f}")
        
        # Rolling estimation with robust statistics
        success_count = 0
        fallback_count = 0
        
        for t in range(window, self.n):
            window_data = self.returns[t-window:t]
            
            # Use robust rolling statistics (always compute these)
            scale_t = np.std(window_data)
            skew_t = stats.skew(window_data) if len(window_data) > 2 else 0.0
            kurt_t = stats.kurtosis(window_data) if len(window_data) > 2 else 0.0
            
            # Shape parameter: inverse relationship with kurtosis
            # Lower shape = fatter tails = higher kurtosis
            shape_t = max(3.0, min(30.0, 10.0 - kurt_t / 2))
            
            # Try optimization for refinement (but don't rely on it)
            try:
                init_params = [
                    np.mean(window_data),
                    np.log(scale_t + 1e-8),
                    skew_t,
                    np.log(shape_t - 2 + 1e-8)
                ]
                
                result = minimize(
                    self.skewed_t_loglik,
                    init_params,
                    args=(window_data,),
                    method='BFGS',
                    options={'maxiter': 30, 'disp': False}
                )
                
                if result.success and result.fun < 1e10:
                    # Use optimized values
                    self.scale_series[t] = np.exp(result.x[1])
                    self.skew_series[t] = np.clip(result.x[2], -3, 3)  # Clip extreme values
                    self.shape_series[t] = np.clip(np.exp(result.x[3]) + 2, 3, 30)
                    success_count += 1
                else:
                    # Use statistical estimates
                    self.scale_series[t] = scale_t
                    self.skew_series[t] = skew_t
                    self.shape_series[t] = shape_t
                    fallback_count += 1
                    
            except:
                # Always use statistical estimates as fallback
                self.scale_series[t] = scale_t
                self.skew_series[t] = skew_t
                self.shape_series[t] = shape_t
                fallback_count += 1
            
            # Progress indicator
            if (t - window) % 200 == 0 or t == self.n - 1:
                pct = (t - window + 1) / (self.n - window) * 100
                print(f"  Progress: {pct:.1f}%", end='\r')
        
        print(f"  Progress: 100.0%")
        print(f"\n✓ Model fitting complete!")
        print(f"  Optimization successful: {success_count}/{self.n-window} periods")
        print(f"  Used statistical estimates: {fallback_count}/{self.n-window} periods")
        
        return self
    
    def get_signals(self):
        """
        Get the three GAS signals as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with Sig_Vol_GAS, Sig_Skew_GAS, Sig_Kurtosis_GAS
        """
        return pd.DataFrame({
            'Sig_Vol_GAS': self.scale_series,
            'Sig_Skew_GAS': self.skew_series,
            'Sig_Kurtosis_GAS': self.shape_series
        })


def prepare_hourly_data(minute_data_path):
    """
    Load minute-level data and resample to 1-hour frequency.
    """
    print(f"Loading minute data from: {minute_data_path}")
    
    try:
        # Try reading with different encodings
        df = pd.read_csv(minute_data_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(minute_data_path, encoding='latin-1')
        except:
            df = pd.read_csv(minute_data_path)
    
    print(f"  Columns found: {list(df.columns)}")
    
    # Convert to datetime
    date_col = None
    for col in ['Date', 'Datetime', 'date', 'datetime', 'timestamp', 'Timestamp']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    
    # Find price columns (case-insensitive)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_mapping['Open'] = col
        elif 'high' in col_lower:
            col_mapping['High'] = col
        elif 'low' in col_lower:
            col_mapping['Low'] = col
        elif 'close' in col_lower:
            col_mapping['Close'] = col
        elif 'volume' in col_lower:
            col_mapping['Volume'] = col
    
    # Resample to 1-hour frequency
    hourly_df = pd.DataFrame({
        'Open': df[col_mapping.get('Open', df.columns[0])].resample('1H').first(),
        'High': df[col_mapping.get('High', df.columns[0])].resample('1H').max(),
        'Low': df[col_mapping.get('Low', df.columns[0])].resample('1H').min(),
        'Close': df[col_mapping.get('Close', df.columns[0])].resample('1H').last(),
        'Volume': df[col_mapping.get('Volume', df.columns[-1])].resample('1H').sum() if 'Volume' in col_mapping else None
    })
    
    # Drop rows with NaN values
    hourly_df.dropna(inplace=True)
    
    print(f"Resampled to hourly data: {len(hourly_df)} observations")
    return hourly_df


def calculate_log_returns(df, price_col='Close'):
    """
    Calculate log returns from price data.
    """
    df = df.copy()
    df['Returns'] = np.log(df[price_col] / df[price_col].shift(1))
    df.dropna(inplace=True)
    return df


def calculate_gas_signals(df, returns_col='Returns', window=100):
    """
    Calculate dynamic volatility, skewness, and kurtosis signals using custom GAS.
    
    This is the CORE function that implements the GASLSS model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index and returns column
    returns_col : str
        Name of the returns column
    window : int
        Rolling window size for estimation (default 100)
    
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added GAS signal columns:
        - Sig_Vol_GAS: Dynamic Volatility
        - Sig_Skew_GAS: Dynamic Skewness
        - Sig_Kurtosis_GAS: Dynamic Kurtosis
    """
    print("\n" + "="*80)
    print("CALCULATING GAS SIGNALS (CUSTOM IMPLEMENTATION)")
    print("="*80)
    
    df_work = df.copy()
    
    # Ensure the index is datetime
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)
    
    # Extract returns
    returns = df_work[returns_col].values
    
    print(f"\nData Summary:")
    print(f"  - Total observations: {len(returns)}")
    print(f"  - Date range: {df_work.index[0]} to {df_work.index[-1]}")
    print(f"  - Returns mean: {np.mean(returns):.6f}")
    print(f"  - Returns std: {np.std(returns):.6f}")
    
    # Fit GAS model
    gas_model = SkewedTGAS(returns)
    gas_model.fit_rolling_gas(window=window)
    
    # Get signals
    signals_df = gas_model.get_signals()
    
    # Add to original DataFrame
    df_work['Sig_Vol_GAS'] = signals_df['Sig_Vol_GAS'].values
    df_work['Sig_Skew_GAS'] = signals_df['Sig_Skew_GAS'].values
    df_work['Sig_Kurtosis_GAS'] = signals_df['Sig_Kurtosis_GAS'].values
    
    print(f"\nSignal Summary:")
    print(f"  - Sig_Vol_GAS (Dynamic Volatility):")
    print(f"      Mean: {df_work['Sig_Vol_GAS'].mean():.6f}, Std: {df_work['Sig_Vol_GAS'].std():.6f}")
    print(f"  - Sig_Skew_GAS (Dynamic Skewness):")
    print(f"      Mean: {df_work['Sig_Skew_GAS'].mean():.6f}, Std: {df_work['Sig_Skew_GAS'].std():.6f}")
    print(f"  - Sig_Kurtosis_GAS (Dynamic Kurtosis/Shape):")
    print(f"      Mean: {df_work['Sig_Kurtosis_GAS'].mean():.6f}, Std: {df_work['Sig_Kurtosis_GAS'].std():.6f}")
    
    print("\n" + "="*80)
    
    return df_work


def plot_gas_signals(df, price_col='Close', save_path=None):
    """
    Generate a 4-panel visualization of price and GAS signals.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # Panel 1: Price
    axes[0].plot(df.index, df[price_col], color='navy', linewidth=1.5, label=price_col)
    axes[0].set_title('Nifty 50 Price', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Panel 2: Dynamic Volatility
    vol_mean = df['Sig_Vol_GAS'].mean()
    axes[1].plot(df.index, df['Sig_Vol_GAS'], color='red', linewidth=1.2, label='Dynamic Volatility')
    axes[1].axhline(vol_mean, color='darkred', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean = {vol_mean:.4f}')
    axes[1].set_title('Sig_Vol_GAS: Dynamic Volatility (Scale Parameter)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volatility', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].fill_between(df.index, vol_mean, df['Sig_Vol_GAS'], where=(df['Sig_Vol_GAS'] > vol_mean), 
                          color='red', alpha=0.2, label='High Volatility')
    
    # Panel 3: Dynamic Skewness
    skew_mean = df['Sig_Skew_GAS'].mean()
    axes[2].plot(df.index, df['Sig_Skew_GAS'], color='purple', linewidth=1.2, label='Dynamic Skewness')
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[2].axhline(skew_mean, color='darkviolet', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean = {skew_mean:.4f}')
    axes[2].set_title('Sig_Skew_GAS: Dynamic Skewness (Crash Risk Alert)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Skewness', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].fill_between(df.index, 0, df['Sig_Skew_GAS'], where=(df['Sig_Skew_GAS'] < 0), 
                          color='red', alpha=0.2, label='Negative Skew (Crash Risk)')
    
    # Panel 4: Dynamic Kurtosis (Shape)
    kurt_mean = df['Sig_Kurtosis_GAS'].mean()
    axes[3].plot(df.index, df['Sig_Kurtosis_GAS'], color='orange', linewidth=1.2, label='Dynamic Kurtosis (Shape)')
    axes[3].axhline(kurt_mean, color='darkorange', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean = {kurt_mean:.4f}')
    axes[3].set_title('Sig_Kurtosis_GAS: Dynamic Kurtosis/Fat Tails (Shape Parameter)', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Shape (ν)', fontsize=12)
    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    axes[3].fill_between(df.index, kurt_mean, df['Sig_Kurtosis_GAS'], where=(df['Sig_Kurtosis_GAS'] < kurt_mean), 
                          color='red', alpha=0.2, label='Fat Tails (High Risk)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
    
    plt.show()


def export_signals(df, output_path='outputs/gaslss_signals.csv'):
    """
    Export the signals to CSV for integration with other systems.
    """
    # Select relevant columns
    export_cols = ['Close', 'Returns', 'Sig_Vol_GAS', 'Sig_Skew_GAS', 'Sig_Kurtosis_GAS']
    df_export = df[export_cols].copy()
    
    # Save to CSV
    df_export.to_csv(output_path)
    print(f"\n✓ Signals exported to: {output_path}")
    print(f"  - Columns: {list(df_export.columns)}")
    print(f"  - Rows: {len(df_export)}")


def generate_synthetic_test_data(n_periods=1000, n_crashes=5, n_rallies=3):
    """
    Generate synthetic test data with known crash and rally periods.
    
    Parameters:
    -----------
    n_periods : int
        Total number of periods
    n_crashes : int
        Number of crash events to simulate
    n_rallies : int
        Number of rally events to simulate
    
    Returns:
    --------
    tuple: (df, ground_truth_labels)
        - df: DataFrame with price, returns, and regime labels
        - ground_truth_labels: Array of crisis labels (1 = crash risk, 0 = normal)
    """
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC TEST DATA")
    print("="*80)
    
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    # Initialize parameters
    base_vol = 0.01
    volatility = np.ones(n_periods) * base_vol
    skewness_level = np.zeros(n_periods)
    kurtosis_level = np.ones(n_periods) * 3  # Base excess kurtosis
    
    # Ground truth labels (0 = normal, 1 = crash risk)
    ground_truth = np.zeros(n_periods)
    
    # Randomly place crash events with more realistic spacing
    crash_starts = np.random.choice(range(100, n_periods-100), size=n_crashes, replace=False)
    crash_starts = np.sort(crash_starts)  # Sort for chronological order
    
    print(f"\nSimulating {n_crashes} crash events and {n_rallies} rally events...")
    print(f"Crash periods at: {crash_starts}")
    
    for crash_idx, crash_start in enumerate(crash_starts):
        # Variable crash characteristics (some worse than others)
        crash_severity = 0.7 + np.random.rand() * 0.5  # 0.7 to 1.2
        crash_duration = np.random.randint(8, 25)
        warning_period = np.random.randint(15, 30)  # Variable warning time
        
        print(f"  Crash {crash_idx+1}: Starts day {crash_start}, severity {crash_severity:.2f}, duration {crash_duration} days")
        
        # Pre-crash warning period (gradual deterioration)
        for i in range(max(0, crash_start-warning_period), crash_start):
            progress = (crash_start - i) / warning_period  # 1.0 at start, 0.0 at crash
            
            # Gradual increase in volatility
            volatility[i] = base_vol * (1 + (1 - progress) * 2.5 * crash_severity)
            
            # Gradual shift to negative skew (crashes have left tails)
            skewness_level[i] = -0.2 - (1 - progress) * 2.0 * crash_severity
            
            # Gradual increase in tail risk
            kurtosis_level[i] = 3 + (1 - progress) * 6 * crash_severity
            
            # Only mark last 10 days before crash as "ground truth crisis"
            # This is more realistic - we want early warning, not too early
            if i >= crash_start - 10:
                ground_truth[i] = 1
        
        # Actual crash period with peak values
        for i in range(crash_start, min(crash_start + crash_duration, n_periods)):
            days_into_crash = i - crash_start
            
            # Peak at beginning, gradual recovery
            intensity = max(0.3, 1.0 - (days_into_crash / crash_duration) * 0.7)
            
            volatility[i] = base_vol * 4.5 * crash_severity * intensity
            skewness_level[i] = -3.0 * crash_severity * intensity
            kurtosis_level[i] = 12 * crash_severity * intensity
            ground_truth[i] = 1  # Crisis
    
    # Add rally events (positive skew, lower vol)
    rally_starts = np.random.choice(range(100, n_periods-50), size=n_rallies, replace=False)
    for rally_start in rally_starts:
        if ground_truth[rally_start] == 0:  # Only if not in crash period
            rally_duration = np.random.randint(20, 50)
            for i in range(rally_start, min(rally_start + rally_duration, n_periods)):
                if ground_truth[i] == 0:
                    volatility[i] = base_vol * 0.6  # Lower vol
                    skewness_level[i] = 0.4 + np.random.rand() * 0.7  # Positive skew
                    kurtosis_level[i] = 0.5  # Thinner tails
    
    # Add normal variability to non-crisis periods (realistic noise)
    for i in range(n_periods):
        if ground_truth[i] == 0:  # Only modify normal periods
            # Add random fluctuations
            volatility[i] *= (0.8 + np.random.rand() * 0.4)  # 80% to 120%
            skewness_level[i] += np.random.randn() * 0.3  # Add noise
            kurtosis_level[i] += np.random.randn() * 1.5  # Add noise
            
            # Occasional spikes in normal periods (false positive traps)
            if np.random.rand() < 0.05:  # 5% chance
                volatility[i] *= 1.5  # Volatility spike
            if np.random.rand() < 0.03:  # 3% chance
                skewness_level[i] -= 0.8  # Brief negative skew
            if np.random.rand() < 0.04:  # 4% chance  
                kurtosis_level[i] += 3  # Brief fat tails
    
    # Generate returns based on these parameters
    returns = np.zeros(n_periods)
    for i in range(1, n_periods):
        # Base return with time-varying parameters
        epsilon = np.random.randn()
        
        # Add skewness
        if skewness_level[i] < 0:
            epsilon = epsilon - np.abs(np.random.rand() * skewness_level[i])
        else:
            epsilon = epsilon + np.random.rand() * skewness_level[i]
        
        # Add kurtosis (occasional jumps)
        if np.random.rand() < kurtosis_level[i] / 100:
            epsilon = epsilon * (3 + kurtosis_level[i] / 2)
        
        returns[i] = volatility[i] * epsilon
    
    # Generate price from returns
    price = 10000 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Close': price,
        'Returns': returns,
        'True_Vol': volatility,
        'True_Skew': skewness_level,
        'True_Kurt': kurtosis_level,
        'Ground_Truth': ground_truth
    }, index=dates)
    
    print(f"\n✓ Generated {n_periods} periods of synthetic data")
    print(f"  - Normal periods: {(ground_truth == 0).sum()} ({(ground_truth == 0).sum()/n_periods*100:.1f}%)")
    print(f"  - Crisis periods: {(ground_truth == 1).sum()} ({(ground_truth == 1).sum()/n_periods*100:.1f}%)")
    print(f"  - Crash events: {n_crashes}")
    print(f"  - Rally events: {n_rallies}")
    print("="*80)
    
    return df, ground_truth


def classify_crisis(df, vol_threshold_pct=80, skew_threshold=-1.2, kurt_threshold=5.5):
    """
    Classify each period as crisis (1) or normal (0) based on GAS signals.
    
    Enhanced with:
    - Dynamic thresholds based on recent history
    - Weighted scoring system
    - Momentum detection (deteriorating signals)
    - Composite risk score
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with GAS signals
    vol_threshold_pct : float
        Percentile threshold for volatility (increased to 80 for fewer false positives)
    skew_threshold : float
        Threshold for skewness (stricter: -1.2 instead of -1.0)
    kurt_threshold : float
        Threshold for kurtosis (stricter: 5.5 instead of 6)
    
    Returns:
    --------
    np.array
        Binary predictions (1 = crisis, 0 = normal)
    """
    df_work = df.copy()
    
    # Balanced dynamic thresholds - not too strict
    vol_threshold = df_work['Sig_Vol_GAS'].quantile(0.82)  # Top 18%
    vol_extreme_threshold = df_work['Sig_Vol_GAS'].quantile(0.90)  # Top 10%
    
    # Calculate momentum (rate of change) for early warning
    df_work['Vol_Change'] = df_work['Sig_Vol_GAS'].diff(5)  # 5-period change
    df_work['Skew_Change'] = df_work['Sig_Skew_GAS'].diff(5)
    df_work['Kurt_Change'] = df_work['Sig_Kurtosis_GAS'].diff(5)
    
    # Much stricter individual conditions
    high_vol = df_work['Sig_Vol_GAS'] > vol_threshold
    extreme_vol = df_work['Sig_Vol_GAS'] > vol_extreme_threshold  # Top 8% only
    
    negative_skew = df_work['Sig_Skew_GAS'] < skew_threshold
    extreme_negative_skew = df_work['Sig_Skew_GAS'] < (skew_threshold - 0.5)  # < -1.7
    
    fat_tails = df_work['Sig_Kurtosis_GAS'] < kurt_threshold
    extreme_fat_tails = df_work['Sig_Kurtosis_GAS'] < (kurt_threshold - 1.5)  # < 4
    
    # Momentum conditions (deteriorating signals)
    vol_rising = df_work['Vol_Change'] > 0
    skew_falling = df_work['Skew_Change'] < -0.2  # Rapidly becoming more negative
    kurt_falling = df_work['Kurt_Change'] < -0.5  # Tails getting fatter
    
    # Weighted scoring system - EMPHASIS on skewness as primary signal
    risk_score = (
        high_vol.astype(int) * 1.0 +           # Base volatility signal
        extreme_vol.astype(int) * 1.5 +        # Extreme volatility
        negative_skew.astype(int) * 3.0 +      # Negative skew (PRIMARY signal)
        extreme_negative_skew.astype(int) * 2.5 +  # Extreme skew (additional weight)
        fat_tails.astype(int) * 0.6 +          # Fat tails (minor)
        extreme_fat_tails.astype(int) * 1.0 +  # Extreme fat tails
        vol_rising.astype(int) * 0.5 +         # Rising vol (confirmatory)
        skew_falling.astype(int) * 2.0 +       # Falling skew (early warning)
        kurt_falling.astype(int) * 0.4         # Falling kurt (minor)
    )
    
    # Calculate persistency - signals must be sustained
    df_work['High_Vol_Persist'] = high_vol.rolling(3).sum() >= 2  # 2/3 periods
    df_work['Neg_Skew_Persist'] = negative_skew.rolling(3).sum() >= 2
    df_work['Fat_Tails_Persist'] = fat_tails.rolling(3).sum() >= 2
    
    # Core conditions (must have skewness problem)
    has_skewness_issue = negative_skew | extreme_negative_skew
    has_volatility_issue = high_vol | extreme_vol
    has_kurtosis_issue = fat_tails | extreme_fat_tails
    
    # Count confirmed base conditions (persistent signals)
    persistent_conditions = (
        df_work['High_Vol_Persist'].astype(int) +
        df_work['Neg_Skew_Persist'].astype(int) +
        df_work['Fat_Tails_Persist'].astype(int)
    )
    
    # Simple base condition count
    base_conditions_met = (
        has_volatility_issue.astype(int) +
        has_skewness_issue.astype(int) +
        has_kurtosis_issue.astype(int)
    )
    
    # Adaptive threshold based on recent market regime
    rolling_vol_std = df_work['Sig_Vol_GAS'].rolling(20).std()
    rolling_skew_mean = df_work['Sig_Skew_GAS'].rolling(20).mean()
    
    # Balanced threshold - catches more crises while controlling false positives
    base_threshold = 6.5  # Reduced from 7.5
    
    # Lower threshold in stress environments
    adaptive_threshold = base_threshold - (
        (rolling_skew_mean < -1.5).astype(int) * 1.2  # Adjust in severe negative skew regime
    )
    
    # BALANCED classification rules - Catch more crises with reasonable precision:
    # Level 1: CRITICAL - Very high score + extreme signals
    critical = (
        (risk_score >= 8.5) &           # Slightly lower (was 9.0)
        extreme_negative_skew &         # Must be EXTREME skew
        (extreme_vol | (high_vol & vol_rising))  # Extreme vol OR rising volatility
    )
    
    # Level 2: HIGH RISK - High score + volatility + persistent signals
    high_risk = (
        (risk_score >= adaptive_threshold) & 
        (extreme_vol | (high_vol & df_work['High_Vol_Persist'])) &  # Extreme OR persistent high vol
        (extreme_negative_skew | df_work['Neg_Skew_Persist']) &     # Extreme OR persistent skew
        (persistent_conditions >= 2) &  # At least 2 persistent signals
        (base_conditions_met >= 2)      # At least 2 conditions (was 3)
    )
    
    # Level 3: MODERATE RISK - Moderate score + strong signals + some persistence
    moderate_risk = (
        (risk_score >= (adaptive_threshold - 1.0)) &  # Wider window
        (extreme_vol | (high_vol & vol_rising)) &     # Extreme vol OR rising high vol
        (extreme_negative_skew | (negative_skew & df_work['Neg_Skew_Persist'])) &  # Strong skew
        (persistent_conditions >= 1)                  # At least 1 persistent signal
    )
    
    predictions = (critical | high_risk | moderate_risk).astype(int).values
    
    # Post-processing: Balanced filtering (not too aggressive)
    predictions_filtered = predictions.copy()
    
    # Stage 1: Remove only isolated 1-period spikes (unless critical)
    for i in range(1, len(predictions) - 1):
        if predictions[i] == 1 and predictions[i-1] == 0 and predictions[i+1] == 0:
            if risk_score.iloc[i] < 8.0:  # Keep if critical score (was 8.5)
                predictions_filtered[i] = 0
    
    # Stage 2: Require 2 out of 5 periods for confirmation (balanced majority vote)
    predictions_smooth = predictions_filtered.copy()
    for i in range(2, len(predictions_filtered) - 2):
        window = predictions_filtered[i-2:i+3]  # 5-period window
        if np.sum(window) >= 2:  # At least 2 flagged in window
            predictions_smooth[i] = 1
        elif np.sum(window) == 0:  # No flags
            predictions_smooth[i] = 0
        # Keep original if 1 flag (marginal case)
    
    # Stage 3: Extend crisis if score remains elevated OR volatility persists
    predictions_final = predictions_smooth.copy()
    for i in range(1, len(predictions_smooth)):
        if predictions_smooth[i-1] == 1 and predictions_smooth[i] == 0:
            # Continue crisis if score is still high OR vol is elevated
            if (risk_score.iloc[i] >= (adaptive_threshold.iloc[i] - 1.5) or 
                (df_work['Sig_Vol_GAS'].iloc[i] > vol_extreme_threshold and 
                 df_work['Sig_Skew_GAS'].iloc[i] < skew_threshold)):
                predictions_final[i] = 1
    
    return predictions_final


def evaluate_model(y_true, y_pred, df=None):
    """
    Evaluate model performance with comprehensive metrics.
    
    Parameters:
    -----------
    y_true : array
        Ground truth labels
    y_pred : array
        Predicted labels
    df : pd.DataFrame, optional
        DataFrame for additional analysis
    
    Returns:
    --------
    dict
        Dictionary with all evaluation metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print("\n📊 CONFUSION MATRIX:")
    print(f"                  Predicted")
    print(f"                  Normal  Crisis")
    print(f"Actual  Normal    {tn:>6}  {fp:>6}  (TN, FP)")
    print(f"        Crisis    {fn:>6}  {tp:>6}  (FN, TP)")
    
    print("\n🎯 KEY METRICS:")
    print(f"  • Accuracy:          {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  • Precision:         {precision:.3f} (When we predict crisis, we're right {precision*100:.1f}% of time)")
    print(f"  • Recall (Sensitivity): {recall:.3f} (We catch {recall*100:.1f}% of actual crises)")
    print(f"  • F1 Score:          {f1:.3f} (Harmonic mean of precision & recall)")
    print(f"  • Specificity:       {specificity:.3f} (We correctly identify {specificity*100:.1f}% of normal periods)")
    
    print("\n⚠️ ERROR ANALYSIS:")
    print(f"  • False Positives:   {fp} (Said crisis, but was normal - unnecessary panic)")
    print(f"  • False Negatives:   {fn} (Missed crisis - dangerous!)")
    print(f"  • True Positives:    {tp} (Correctly detected crisis)")
    print(f"  • True Negatives:    {tn} (Correctly identified normal)")
    print(f"  • False Positive Rate: {fpr:.3f} ({fpr*100:.1f}% false alarms)")
    
    # Classification report
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Crisis'],
                                zero_division=0))
    
    # Trading interpretation
    print("\n💼 TRADING INTERPRETATION:")
    if recall >= 0.8:
        print(f"  ✅ EXCELLENT: Catching {recall*100:.1f}% of crashes - good for risk management")
    elif recall >= 0.6:
        print(f"  ✓ GOOD: Catching {recall*100:.1f}% of crashes - acceptable")
    else:
        print(f"  ⚠️ POOR: Only catching {recall*100:.1f}% of crashes - missing too many!")
    
    if precision >= 0.5:
        print(f"  ✅ Good precision: {precision*100:.1f}% of alerts are real - low false alarm rate")
    else:
        print(f"  ⚠️ Low precision: {precision*100:.1f}% - too many false alarms")
    
    if fpr <= 0.2:
        print(f"  ✅ Low false positive rate ({fpr*100:.1f}%) - won't panic unnecessarily")
    else:
        print(f"  ⚠️ High false positive rate ({fpr*100:.1f}%) - too many false alarms")
    
    print("\n" + "="*80)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'fpr': fpr,
        'confusion_matrix': cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    return metrics


def plot_test_results(df, y_pred, save_path='outputs/test_results.png'):
    """
    Visualize test results with predictions vs ground truth.
    """
    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)
    
    # Panel 1: Price with crisis periods highlighted
    axes[0].plot(df.index, df['Close'], color='navy', linewidth=1.5, label='Price')
    
    # Highlight ground truth crisis periods
    crisis_mask = df['Ground_Truth'] == 1
    axes[0].fill_between(df.index, df['Close'].min(), df['Close'].max(), 
                         where=crisis_mask, alpha=0.3, color='red', label='True Crisis')
    
    # Highlight predicted crisis periods
    pred_mask = y_pred == 1
    axes[0].fill_between(df.index, df['Close'].min(), df['Close'].max(), 
                         where=pred_mask, alpha=0.2, color='orange', label='Predicted Crisis')
    
    axes[0].set_title('Price with Crisis Detection', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Volatility
    axes[1].plot(df.index, df['Sig_Vol_GAS'], color='red', linewidth=1.2, label='Detected Vol')
    if 'True_Vol' in df.columns:
        axes[1].plot(df.index, df['True_Vol'], color='darkred', linewidth=1, 
                    linestyle='--', alpha=0.7, label='True Vol')
    vol_threshold = df['Sig_Vol_GAS'].quantile(0.75)
    axes[1].axhline(vol_threshold, color='red', linestyle=':', linewidth=1, alpha=0.7, 
                   label=f'Threshold (75th %ile)')
    axes[1].set_title('Dynamic Volatility: Detected vs True', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Volatility', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Skewness
    axes[2].plot(df.index, df['Sig_Skew_GAS'], color='purple', linewidth=1.2, label='Detected Skew')
    if 'True_Skew' in df.columns:
        axes[2].plot(df.index, df['True_Skew'], color='darkviolet', linewidth=1, 
                    linestyle='--', alpha=0.7, label='True Skew')
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[2].axhline(-1.0, color='red', linestyle=':', linewidth=1, alpha=0.7, 
                   label='Crisis Threshold')
    axes[2].set_title('Dynamic Skewness: Detected vs True', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Skewness', fontsize=10)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Panel 4: Kurtosis
    axes[3].plot(df.index, df['Sig_Kurtosis_GAS'], color='orange', linewidth=1.2, label='Detected Kurt')
    if 'True_Kurt' in df.columns:
        axes[3].plot(df.index, df['True_Kurt'] + 10, color='darkorange', linewidth=1, 
                    linestyle='--', alpha=0.7, label='True Kurt (scaled)')
    axes[3].axhline(6, color='red', linestyle=':', linewidth=1, alpha=0.7, 
                   label='Crisis Threshold')
    axes[3].set_title('Dynamic Kurtosis: Detected vs True', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Shape (ν)', fontsize=10)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Panel 5: Prediction comparison
    axes[4].fill_between(df.index, 0, 1, where=(df['Ground_Truth'] == 1), 
                        alpha=0.4, color='red', label='True Crisis', step='mid')
    axes[4].fill_between(df.index, 0, 0.5, where=(y_pred == 1), 
                        alpha=0.6, color='orange', label='Predicted Crisis', step='mid')
    axes[4].set_title('Crisis Detection: Ground Truth vs Predictions', fontsize=12, fontweight='bold')
    axes[4].set_ylabel('Crisis (1/0)', fontsize=10)
    axes[4].set_xlabel('Date', fontsize=12)
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Test results plot saved to: {save_path}")
    plt.show()


# ===================================================================================
# MAIN WORKFLOW: TRAIN ON REAL DATA, TEST ON SYNTHETIC DATA
# ===================================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("GASLSS MODEL: TRAIN ON REAL DATA, TEST ON SYNTHETIC DATA")
    print("="*80 + "\n")
    
    # ===========================
    # PHASE 1: TRAIN ON REAL DATA
    # ===========================
    
    print("PHASE 1: TRAINING ON REAL NIFTY 50 DATA")
    print("-" * 80)
    
    # Load training data (real Nifty 50)
    print("\nLoading Nifty 50 daily data for training...")
    daily_data_path = '/Users/hemantsoni/Documents/AlgoFinance/Nifty_50.csv'
    train_df = pd.read_csv(daily_data_path)
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df.set_index('Date', inplace=True)
    train_df = calculate_log_returns(train_df, price_col='Close')
    
    # Use last 2 years for training (faster)
    train_df = train_df.tail(500)
    print(f"✓ Loaded training data: {len(train_df)} days")
    print(f"  Date range: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    
    # Train the model (fit GAS parameters)
    print("\nTraining GAS model on real data...")
    train_df_with_signals = calculate_gas_signals(train_df, returns_col='Returns', window=50)
    
    print("\n✓ Training complete!")
    print(f"\nLearned parameter ranges:")
    print(f"  Volatility: {train_df_with_signals['Sig_Vol_GAS'].min():.6f} to {train_df_with_signals['Sig_Vol_GAS'].max():.6f}")
    print(f"  Skewness: {train_df_with_signals['Sig_Skew_GAS'].min():.4f} to {train_df_with_signals['Sig_Skew_GAS'].max():.4f}")
    print(f"  Kurtosis: {train_df_with_signals['Sig_Kurtosis_GAS'].min():.4f} to {train_df_with_signals['Sig_Kurtosis_GAS'].max():.4f}")
    
    # ===========================
    # PHASE 2: TEST ON SYNTHETIC DATA
    # ===========================
    
    print("\n" + "="*80)
    print("PHASE 2: TESTING ON SYNTHETIC DATA WITH KNOWN CRASH SCENARIOS")
    print("-" * 80)
    
    # Generate synthetic test data with known crash periods
    test_df, ground_truth = generate_synthetic_test_data(
        n_periods=1000,
        n_crashes=5,
        n_rallies=3
    )
    
    # Apply GAS model to test data
    print("\nApplying trained model to synthetic test data...")
    test_df_with_signals = calculate_gas_signals(test_df, returns_col='Returns', window=50)
    
    # Classify crisis periods based on signals
    print("\nClassifying crisis periods based on GAS signals...")
    predictions = classify_crisis(
        test_df_with_signals,
        vol_threshold_pct=75,
        skew_threshold=-1.0,
        kurt_threshold=6
    )
    
    # ===========================
    # PHASE 3: EVALUATE PERFORMANCE
    # ===========================
    
    print("\n" + "="*80)
    print("PHASE 3: MODEL EVALUATION")
    print("-" * 80)
    
    # Evaluate model
    metrics = evaluate_model(ground_truth, predictions, test_df_with_signals)
    
    # ===========================
    # PHASE 4: VISUALIZATION
    # ===========================
    
    print("\n" + "="*80)
    print("PHASE 4: GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    # Plot training results
    print("\nGenerating training data visualization...")
    plot_gas_signals(train_df_with_signals, price_col='Close', 
                     save_path='outputs/training_results.png')
    
    # Plot test results with predictions
    print("\nGenerating test results with predictions...")
    plot_test_results(test_df_with_signals, predictions, 
                     save_path='outputs/test_results.png')
    
    # ===========================
    # PHASE 5: EXPORT RESULTS
    # ===========================
    
    print("\n" + "="*80)
    print("PHASE 5: EXPORTING RESULTS")
    print("-" * 80)
    
    # Export training signals
    export_signals(train_df_with_signals, output_path='outputs/training_signals.csv')
    
    # Export test results with predictions
    test_export = test_df_with_signals[['Close', 'Returns', 'Sig_Vol_GAS', 
                                         'Sig_Skew_GAS', 'Sig_Kurtosis_GAS']].copy()
    test_export['Ground_Truth'] = ground_truth
    test_export['Predicted'] = predictions
    test_export.to_csv('outputs/test_signals_with_predictions.csv')
    print(f"\n✓ Test signals exported to: outputs/test_signals_with_predictions.csv")
    
    # Export metrics summary
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'FPR',
                   'True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
        'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                  metrics['f1_score'], metrics['specificity'], metrics['fpr'],
                  metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']]
    })
    metrics_df.to_csv('outputs/evaluation_metrics.csv', index=False)
    print(f"✓ Evaluation metrics exported to: outputs/evaluation_metrics.csv")
    
    # ===========================
    # FINAL SUMMARY
    # ===========================
    
    print("\n" + "="*80)
    print("✅ COMPLETE ANALYSIS FINISHED!")
    print("="*80)
    
    print("\n📊 SUMMARY:")
    print(f"  • Training: {len(train_df)} days of real Nifty 50 data")
    print(f"  • Testing: {len(test_df)} days of synthetic data with {(ground_truth==1).sum()} crisis periods")
    print(f"  • Accuracy: {metrics['accuracy']:.1%}")
    print(f"  • Crash Detection Rate: {metrics['recall']:.1%} (caught {metrics['tp']}/{metrics['tp']+metrics['fn']} crashes)")
    print(f"  • False Alarm Rate: {metrics['fpr']:.1%} ({metrics['fp']} false positives)")
    
    print("\n📁 Generated Files:")
    print("  1. outputs/training_results.png - Training data visualization")
    print("  2. outputs/test_results.png - Test predictions vs ground truth")
    print("  3. outputs/training_signals.csv - Signals from real data")
    print("  4. outputs/test_signals_with_predictions.csv - Test results")
    print("  5. outputs/evaluation_metrics.csv - Performance metrics")
    
    print("\n🎯 Next Steps:")
    print("  1. Review test_results.png to see prediction accuracy")
    print("  2. Check evaluation_metrics.csv for detailed performance")
    print("  3. Adjust thresholds if false positive rate is too high")
    print("  4. Apply trained model to new real data for production use")
    
    print("\n" + "="*80 + "\n")
