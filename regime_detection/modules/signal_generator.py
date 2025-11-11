"""
Signal Generator Module
Converts changepoint dates into a binary time series signal for regime identification
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeSignalGenerator:
    """
    Generates a binary signal from detected changepoints.
    
    The signal is:
    - 0 on normal days (no regime change)
    - 1 on days when a regime change is detected
    
    This signal can be used by downstream models or trading systems.
    """
    
    def __init__(self, date_index: pd.DatetimeIndex):
        """
        Initialize the RegimeSignalGenerator.
        
        Args:
            date_index (pd.DatetimeIndex): Full date range for the signal
        """
        self.date_index = date_index
        self.signal = None
        self.regime_labels = None
        
    def generate_binary_signal(self, changepoint_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate a binary signal (0/1) where 1 indicates a regime change.
        
        Args:
            changepoint_dates (pd.DatetimeIndex): Dates where changepoints detected
            
        Returns:
            pd.Series: Binary signal with date index
        """
        logger.info(f"Generating binary signal for {len(changepoint_dates)} changepoints")
        
        # Initialize signal with zeros
        signal = pd.Series(0, index=self.date_index, name='regime_change_signal')
        
        # Set 1 at changepoint dates
        for date in changepoint_dates:
            if date in signal.index:
                signal.loc[date] = 1
            else:
                # Find nearest date if exact match not found
                nearest_date = signal.index[signal.index.get_indexer([date], method='nearest')[0]]
                signal.loc[nearest_date] = 1
                logger.debug(f"Changepoint {date} mapped to nearest date {nearest_date}")
        
        self.signal = signal
        
        logger.info(f"✓ Generated binary signal: {signal.sum()} regime changes detected")
        
        return signal
    
    def generate_regime_labels(self, changepoint_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate regime labels (0, 1, 2, ...) for each regime period.
        
        Each period between changepoints gets a unique regime ID.
        This is useful for regime-based analysis.
        
        Args:
            changepoint_dates (pd.DatetimeIndex): Dates where changepoints detected
            
        Returns:
            pd.Series: Regime labels with date index
        """
        logger.info(f"Generating regime labels for {len(changepoint_dates) + 1} regimes")
        
        # Initialize labels with zeros
        labels = pd.Series(0, index=self.date_index, name='regime_label')
        
        # Sort changepoint dates
        sorted_dates = sorted(changepoint_dates)
        
        # Assign regime labels
        current_regime = 0
        for i, date in enumerate(sorted_dates):
            # Find the date in the index
            if date in labels.index:
                change_idx = labels.index.get_loc(date)
            else:
                change_idx = labels.index.get_indexer([date], method='nearest')[0]
            
            # Increment regime for all dates after this changepoint
            labels.iloc[change_idx:] = i + 1
            current_regime = i + 1
        
        self.regime_labels = labels
        
        logger.info(f"✓ Generated regime labels: {current_regime + 1} regimes identified")
        
        return labels
    
    def get_regime_statistics(self, data: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
        """
        Calculate statistics for each regime period.
        
        Args:
            data (pd.DataFrame): Price data
            price_column (str): Column name for price
            
        Returns:
            pd.DataFrame: Statistics for each regime
        """
        if self.regime_labels is None:
            raise ValueError("Regime labels not generated. Call generate_regime_labels() first.")
        
        logger.info("Calculating regime statistics")
        
        # Find price column
        price_col = None
        for col in data.columns:
            if col.lower() == price_column.lower():
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"Price column '{price_column}' not found")
        
        # Merge labels with data
        data_with_regimes = data.copy()
        data_with_regimes['regime'] = self.regime_labels
        
        # Calculate returns if not present
        if 'log_returns' not in data_with_regimes.columns:
            data_with_regimes['log_returns'] = np.log(data_with_regimes[price_col] / 
                                                       data_with_regimes[price_col].shift(1))
        
        # Group by regime and calculate statistics
        regime_stats = []
        
        for regime_id in sorted(data_with_regimes['regime'].unique()):
            regime_data = data_with_regimes[data_with_regimes['regime'] == regime_id]
            
            if len(regime_data) > 1:
                start_date = regime_data.index[0]
                end_date = regime_data.index[-1]
                duration_days = (end_date - start_date).days
                
                start_price = regime_data[price_col].iloc[0]
                end_price = regime_data[price_col].iloc[-1]
                total_return = (end_price / start_price - 1) * 100
                
                returns = regime_data['log_returns'].dropna()
                avg_daily_return = returns.mean() * 100
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                
                regime_stats.append({
                    'regime_id': int(regime_id),
                    'start_date': start_date.date(),
                    'end_date': end_date.date(),
                    'duration_days': duration_days,
                    'total_return_pct': round(total_return, 2),
                    'avg_daily_return_pct': round(avg_daily_return, 4),
                    'annualized_volatility_pct': round(volatility, 2),
                    'sharpe_ratio': round(avg_daily_return / (returns.std() * 100) * np.sqrt(252), 2) if returns.std() > 0 else 0
                })
        
        stats_df = pd.DataFrame(regime_stats)
        
        logger.info(f"✓ Calculated statistics for {len(stats_df)} regimes")
        
        return stats_df
    
    def export_signals(self, output_path: str) -> None:
        """
        Export the generated signals to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if self.signal is None:
            raise ValueError("Signal not generated. Call generate_binary_signal() first.")
        
        # Combine signals
        output_df = pd.DataFrame({
            'date': self.signal.index,
            'regime_change_signal': self.signal.values
        })
        
        if self.regime_labels is not None:
            output_df['regime_label'] = self.regime_labels.values
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        logger.info(f"✓ Exported signals to {output_path}")
    
    def get_changepoint_windows(self, 
                               changepoint_dates: pd.DatetimeIndex,
                               window_before: int = 30,
                               window_after: int = 30) -> pd.DataFrame:
        """
        Get time windows around each changepoint for detailed analysis.
        
        Args:
            changepoint_dates (pd.DatetimeIndex): Dates of changepoints
            window_before (int): Days before changepoint
            window_after (int): Days after changepoint
            
        Returns:
            pd.DataFrame: Windows around changepoints
        """
        logger.info(f"Extracting windows around {len(changepoint_dates)} changepoints")
        
        windows = []
        
        for i, cp_date in enumerate(changepoint_dates):
            # Find the position in the index
            try:
                cp_idx = self.date_index.get_loc(cp_date)
            except KeyError:
                cp_idx = self.date_index.get_indexer([cp_date], method='nearest')[0]
            
            # Calculate window boundaries
            start_idx = max(0, cp_idx - window_before)
            end_idx = min(len(self.date_index) - 1, cp_idx + window_after)
            
            windows.append({
                'changepoint_id': i + 1,
                'changepoint_date': cp_date,
                'window_start': self.date_index[start_idx],
                'window_end': self.date_index[end_idx],
                'days_before': cp_idx - start_idx,
                'days_after': end_idx - cp_idx
            })
        
        windows_df = pd.DataFrame(windows)
        
        return windows_df


if __name__ == "__main__":
    # Test the module
    print("Signal Generator module loaded successfully")
