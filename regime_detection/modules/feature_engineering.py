"""
Feature Engineering Module
Calculates stationary features (log returns, volatility) for changepoint detection
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Calculates financial features from OHLCV data for regime detection.
    
    Key principle: Changepoint detection requires STATIONARY signals.
    Raw prices have drift and are non-stationary, so we use:
    1. Log returns (stationary)
    2. Rolling volatility of returns (captures regime changes)
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FeatureEngineer.
        
        Args:
            data (pd.DataFrame): OHLCV data with datetime index
        """
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)
        
    def calculate_log_returns(self, price_column: str = 'Close') -> pd.Series:
        """
        Calculate log returns from price data.
        
        Log returns are preferred over simple returns because:
        - They are time-additive
        - More normally distributed
        - Stationary (no trend)
        
        Args:
            price_column (str): Column name for price (default: 'Close')
            
        Returns:
            pd.Series: Log returns
        """
        logger.info(f"Calculating log returns from {price_column} prices")
        
        # Find the correct column name (case insensitive)
        col_name = None
        for col in self.data.columns:
            if col.lower() == price_column.lower():
                col_name = col
                break
        
        if col_name is None:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        # Calculate log returns: ln(P_t / P_t-1)
        prices = self.data[col_name]
        log_returns = np.log(prices / prices.shift(1))
        
        self.features['log_returns'] = log_returns
        
        # Remove the first NaN value
        valid_returns = log_returns.dropna()
        logger.info(f"Calculated {len(valid_returns)} log returns")
        logger.info(f"Mean: {valid_returns.mean():.6f}, Std: {valid_returns.std():.6f}")
        
        return log_returns
    
    def calculate_rolling_volatility(self, 
                                     window: int = 30,
                                     return_column: str = 'log_returns') -> pd.Series:
        """
        Calculate rolling standard deviation (volatility) of returns.
        
        This is THE KEY SIGNAL for changepoint detection.
        Volatility changes dramatically during regime shifts.
        
        Args:
            window (int): Rolling window size in days (default: 30)
            return_column (str): Column name for returns
            
        Returns:
            pd.Series: Rolling volatility
        """
        logger.info(f"Calculating {window}-day rolling volatility")
        
        # Get returns (calculate if not already done)
        if return_column not in self.features.columns:
            if return_column == 'log_returns':
                self.calculate_log_returns()
            else:
                raise ValueError(f"Return column '{return_column}' not found")
        
        returns = self.features[return_column]
        
        # Calculate rolling standard deviation
        rolling_vol = returns.rolling(window=window).std()
        
        # Annualize the volatility (optional but standard practice)
        # Daily vol * sqrt(252 trading days)
        rolling_vol_annualized = rolling_vol * np.sqrt(252)
        
        self.features[f'rolling_vol_{window}d'] = rolling_vol_annualized
        
        # Remove NaN values for statistics
        valid_vol = rolling_vol_annualized.dropna()
        logger.info(f"Calculated {len(valid_vol)} volatility values")
        logger.info(f"Mean volatility: {valid_vol.mean():.4f}, Min: {valid_vol.min():.4f}, Max: {valid_vol.max():.4f}")
        
        return rolling_vol_annualized
    
    def calculate_multiple_volatilities(self, windows: list = [20, 30, 60]) -> pd.DataFrame:
        """
        Calculate rolling volatilities for multiple window sizes.
        
        Different windows capture different regime characteristics:
        - Short windows (20d): Capture rapid changes
        - Medium windows (30d): Standard measure
        - Long windows (60d): Capture sustained regime changes
        
        Args:
            windows (list): List of window sizes
            
        Returns:
            pd.DataFrame: Multiple volatility columns
        """
        logger.info(f"Calculating volatilities for windows: {windows}")
        
        # Ensure returns are calculated
        if 'log_returns' not in self.features.columns:
            self.calculate_log_returns()
        
        volatilities = {}
        for window in windows:
            vol = self.calculate_rolling_volatility(window=window)
            volatilities[f'vol_{window}d'] = vol
        
        return pd.DataFrame(volatilities, index=self.data.index)
    
    def get_clean_signal(self, signal_name: str) -> tuple:
        """
        Get a clean signal ready for changepoint detection.
        
        Returns the signal as a numpy array with NaN values removed,
        along with the corresponding dates.
        
        Args:
            signal_name (str): Name of the signal column
            
        Returns:
            tuple: (signal_array, dates_array)
        """
        if signal_name not in self.features.columns:
            raise ValueError(f"Signal '{signal_name}' not found. Available: {list(self.features.columns)}")
        
        # Get the signal and remove NaN values
        signal_series = self.features[signal_name].dropna()
        
        signal_array = signal_series.to_numpy().reshape(-1, 1)  # ruptures expects 2D array
        dates_array = signal_series.index.to_numpy()
        
        logger.info(f"Prepared clean signal '{signal_name}' with {len(signal_array)} data points")
        
        return signal_array, dates_array
    
    def get_features_dataframe(self) -> pd.DataFrame:
        """
        Get all calculated features as a DataFrame.
        
        Returns:
            pd.DataFrame: All features
        """
        return self.features
    
    def add_price_data(self) -> pd.DataFrame:
        """
        Combine original price data with calculated features.
        
        Returns:
            pd.DataFrame: Combined data
        """
        combined = pd.concat([self.data, self.features], axis=1)
        return combined


if __name__ == "__main__":
    # Test the module
    from data_loader import DataLoader
    
    loader = DataLoader("../../Nifty_50.csv")
    data = loader.load_data()
    data = loader.clean_data()
    
    engineer = FeatureEngineer(data)
    engineer.calculate_log_returns()
    engineer.calculate_rolling_volatility(window=30)
    
    signal, dates = engineer.get_clean_signal('rolling_vol_30d')
    print(f"\nSignal shape: {signal.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
