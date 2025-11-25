"""
Enhanced Feature Engineering Module for BCD-based Regime Detection
Generates multiple signals optimized for Bayesian Changepoint Detection
Focus: High-quality features that capture regime changes with minimal noise
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """
    Creates robust technical features for regime detection using BCD.
    
    Features are designed to:
    1. Capture volatility regime changes
    2. Detect variance shifts
    3. Minimize false signals
    4. Work well with Bayesian changepoint detection
    """
    
    def __init__(self,
                 vol_window_short: int = 20,
                 vol_window_medium: int = 60,
                 vol_window_long: int = 120,
                 returns_lag: int = 1,
                 volume_window: int = 20,
                 range_window: int = 10):
        """
        Initialize feature engineer with window parameters.
        
        Args:
            vol_window_short: Short-term volatility window (default: 20)
            vol_window_medium: Medium-term volatility window (default: 60)
            vol_window_long: Long-term volatility window (default: 120)
            returns_lag: Lag for returns calculation (default: 1)
            volume_window: Rolling window for volume analysis (default: 20)
            range_window: Window for High-Low range analysis (default: 10)
        """
        self.vol_window_short = vol_window_short
        self.vol_window_medium = vol_window_medium
        self.vol_window_long = vol_window_long
        self.returns_lag = returns_lag
        self.volume_window = volume_window
        self.range_window = range_window
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for BCD regime detection.
        
        Args:
            data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Core price features
        features['close'] = data['Close']
        features['returns'] = data['Close'].pct_change(self.returns_lag)
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(self.returns_lag))
        
        # === VOLATILITY FEATURES (Primary signals for BCD) ===
        
        # 1. Standard rolling volatility (multiple windows)
        features['volatility_20d'] = self._rolling_volatility(data['Close'], self.vol_window_short)
        features['volatility_60d'] = self._rolling_volatility(data['Close'], self.vol_window_medium)
        features['volatility_120d'] = self._rolling_volatility(data['Close'], self.vol_window_long)
        
        # 2. Returns-squared (proxy for variance - excellent for BCD)
        features['returns_squared'] = features['returns'] ** 2
        features['returns_squared_ma'] = features['returns_squared'].rolling(
            window=self.vol_window_short, min_periods=5
        ).mean()
        
        # 3. Absolute returns (robust to direction)
        features['abs_returns'] = features['returns'].abs()
        features['abs_returns_ma'] = features['abs_returns'].rolling(
            window=self.vol_window_short, min_periods=5
        ).mean()
        
        # 4. Volatility of volatility (second-order effect)
        features['vol_of_vol'] = features['volatility_20d'].rolling(
            window=self.vol_window_short, min_periods=5
        ).std()
        
        # 5. Volatility ratio (short/long term - detects regime transitions)
        features['vol_ratio'] = features['volatility_20d'] / (features['volatility_60d'] + 1e-8)
        
        # === VOLUME FEATURES ===
        
        if 'Volume' in data.columns:
            # Volume volatility
            features['volume'] = data['Volume']
            features['volume_returns'] = data['Volume'].pct_change()
            features['volume_volatility'] = self._rolling_volatility(
                data['Volume'], self.volume_window
            )
            
            # Volume momentum
            features['volume_ma'] = data['Volume'].rolling(
                window=self.volume_window, min_periods=5
            ).mean()
            features['volume_ratio'] = data['Volume'] / (features['volume_ma'] + 1)
            
            # Price-Volume correlation (liquidity stress indicator)
            features['price_volume_corr'] = self._rolling_correlation(
                features['abs_returns'], 
                features['volume_returns'].abs(),
                window=self.volume_window
            )
        else:
            # Fill with zeros if volume not available
            features['volume_volatility'] = 0
            features['volume_ratio'] = 1
            features['price_volume_corr'] = 0
        
        # === RANGE-BASED FEATURES ===
        
        if 'High' in data.columns and 'Low' in data.columns:
            # True Range (captures intraday volatility)
            features['true_range'] = self._calculate_true_range(
                data['High'], data['Low'], data['Close']
            )
            
            # Range volatility
            features['range_volatility'] = features['true_range'].rolling(
                window=self.range_window, min_periods=3
            ).std()
            
            # Parkinson volatility (efficient range-based estimator)
            features['parkinson_vol'] = self._parkinson_volatility(
                data['High'], data['Low'], window=self.range_window
            )
            
            # High-Low percentage
            features['hl_pct'] = (data['High'] - data['Low']) / data['Close']
            features['hl_pct_ma'] = features['hl_pct'].rolling(
                window=self.range_window, min_periods=3
            ).mean()
        else:
            # Fill with approximations if OHLC not available
            features['range_volatility'] = features['volatility_20d']
            features['parkinson_vol'] = features['volatility_20d']
            features['hl_pct'] = 0
        
        # === MOMENTUM AND TREND FEATURES ===
        
        # Moving averages
        features['sma_20'] = data['Close'].rolling(window=20, min_periods=5).mean()
        features['sma_50'] = data['Close'].rolling(window=50, min_periods=10).mean()
        
        # Exponential moving averages
        features['ema_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        features['ema_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        # Price position relative to MA
        features['price_to_sma20'] = data['Close'] / (features['sma_20'] + 1e-8)
        features['price_to_sma50'] = data['Close'] / (features['sma_50'] + 1e-8)
        
        # Momentum
        features['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        features['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Rate of change
        features['roc_5'] = data['Close'].pct_change(5)
        features['roc_20'] = data['Close'].pct_change(20)
        
        # === ADVANCED VOLATILITY METRICS ===
        
        # Realized volatility (sum of squared returns)
        features['realized_vol'] = np.sqrt(
            features['returns_squared'].rolling(
                window=self.vol_window_short, min_periods=5
            ).sum()
        )
        
        # Downside volatility (only negative returns)
        downside_returns = features['returns'].copy()
        downside_returns[downside_returns > 0] = 0
        features['downside_vol'] = downside_returns.rolling(
            window=self.vol_window_short, min_periods=5
        ).std()
        
        # Upside volatility (only positive returns)
        upside_returns = features['returns'].copy()
        upside_returns[upside_returns < 0] = 0
        features['upside_vol'] = upside_returns.rolling(
            window=self.vol_window_short, min_periods=5
        ).std()
        
        # Asymmetry ratio
        features['vol_asymmetry'] = features['downside_vol'] / (features['upside_vol'] + 1e-8)
        
        # === REGIME INDICATORS ===
        
        # Volatility state (normalized)
        features['vol_zscore'] = self._zscore(
            features['volatility_20d'], window=self.vol_window_medium
        )
        
        # Volatility percentile (where is current vol in historical distribution)
        features['vol_percentile'] = features['volatility_20d'].rolling(
            window=self.vol_window_long, min_periods=20
        ).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)
        
        # Regime change indicator (large moves in volatility)
        features['vol_change'] = features['volatility_20d'].pct_change()
        features['vol_acceleration'] = features['vol_change'].diff()
        
        # === CLEAN UP ===
        
        # Forward fill then backward fill remaining NaNs
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Replace infinities with NaN and then fill
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Ensure all features are finite
        features = features.clip(-1e10, 1e10)
        
        return features
    
    @staticmethod
    def _rolling_volatility(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling standard deviation (volatility)"""
        returns = series.pct_change()
        vol = returns.rolling(window=window, min_periods=max(5, window//4)).std()
        # Annualize if needed (assuming daily data, multiply by sqrt(252))
        # For hourly data, adjust accordingly
        return vol
    
    @staticmethod
    def _calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range (ATR component)"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    @staticmethod
    def _parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """
        Parkinson volatility estimator (more efficient than close-to-close)
        Uses high-low range information
        """
        hl_ratio = np.log(high / low)
        parkinson = np.sqrt((hl_ratio ** 2).rolling(window=window, min_periods=3).mean() / (4 * np.log(2)))
        return parkinson
    
    @staticmethod
    def _rolling_correlation(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """Calculate rolling correlation between two series"""
        corr = series1.rolling(window=window, min_periods=max(5, window//4)).corr(series2)
        return corr.fillna(0)
    
    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score"""
        mean = series.rolling(window=window, min_periods=max(5, window//4)).mean()
        std = series.rolling(window=window, min_periods=max(5, window//4)).std()
        zscore = (series - mean) / (std + 1e-8)
        return zscore
    
    def get_bcd_signals(self, features: pd.DataFrame) -> List[str]:
        """
        Return list of feature names most suitable for BCD.
        These are the primary signals for changepoint detection.
        
        Returns:
            List of feature column names
        """
        bcd_signals = [
            'volatility_20d',        # Primary signal
            'returns_squared',       # Variance proxy
            'vol_of_vol',           # Second-order effect
            'volume_volatility',     # Volume-based
            'range_volatility',      # Range-based
            'realized_vol',          # Alternative vol measure
            'parkinson_vol',         # Efficient estimator
            'abs_returns_ma',        # Smoothed absolute returns
            'downside_vol',          # Asymmetric risk
            'vol_ratio'              # Regime transition indicator
        ]
        
        # Filter to only include columns that exist in features
        available_signals = [sig for sig in bcd_signals if sig in features.columns]
        
        return available_signals
    
    def validate_features(self, features: pd.DataFrame) -> dict:
        """
        Validate feature quality and provide diagnostics.
        
        Returns:
            Dictionary with validation metrics
        """
        validation = {
            'total_features': len(features.columns),
            'total_rows': len(features),
            'missing_pct': features.isnull().sum().sum() / (len(features) * len(features.columns)) * 100,
            'inf_count': np.isinf(features.values).sum(),
            'zero_variance_features': (features.std() == 0).sum()
        }
        
        # Check key signals
        key_signals = ['volatility_20d', 'returns_squared', 'vol_of_vol']
        validation['key_signals_present'] = sum([sig in features.columns for sig in key_signals])
        
        return validation


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quick_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quick feature generation with default parameters.
    Convenience function for simple use cases.
    """
    engineer = EnhancedFeatureEngineer()
    return engineer.engineer_features(data)


def get_default_bcd_signal(features: pd.DataFrame) -> str:
    """
    Get the default (best) signal for BCD.
    
    Returns:
        Name of the recommended primary signal
    """
    if 'volatility_20d' in features.columns:
        return 'volatility_20d'
    elif 'returns_squared' in features.columns:
        return 'returns_squared'
    elif 'realized_vol' in features.columns:
        return 'realized_vol'
    else:
        raise ValueError("No suitable BCD signal found in features")
