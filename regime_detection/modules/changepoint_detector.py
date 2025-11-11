"""
Changepoint Detection Module
Implements ruptures-based changepoint detection for regime identification
"""

import numpy as np
import pandas as pd
import ruptures as rpt
import logging
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChangepointDetector:
    """
    Detects structural breaks in time series using ruptures library.
    
    Uses the Pelt (Pruned Exact Linear Time) algorithm with L2-norm cost function
    to detect changes in the mean of the volatility signal.
    """
    
    def __init__(self, model: str = "l2", jump: int = 1, min_size: int = 2):
        """
        Initialize the ChangepointDetector.
        
        Args:
            model (str): Cost function model. Options:
                - "l2": Detects changes in mean (recommended for volatility)
                - "rbf": Detects changes using radial basis function
                - "normal": Detects changes in mean and variance
            jump (int): Subsample step (default: 1, no subsampling)
            min_size (int): Minimum segment length between changepoints
        """
        self.model = model
        self.jump = jump
        self.min_size = min_size
        self.algo = None
        self.signal = None  # Store the signal for later reference
        self.breakpoints = None
        self.breakpoint_dates = None
        
        logger.info(f"Initialized ChangepointDetector with model='{model}'")
    
    def fit(self, signal: np.ndarray) -> 'ChangepointDetector':
        """
        Fit the changepoint detection model to the signal.
        
        Args:
            signal (np.ndarray): Input signal (volatility series)
            
        Returns:
            ChangepointDetector: self
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        # Store the signal for later reference
        self.signal = signal
        
        logger.info(f"Fitting changepoint model on signal of length {len(signal)}")
        
        # Initialize and fit the Pelt algorithm
        self.algo = rpt.Pelt(model=self.model, jump=self.jump, min_size=self.min_size)
        self.algo.fit(signal)
        
        logger.info("✓ Model fitted successfully")
        return self
    
    def predict_with_penalty(self, pen: float, signal: np.ndarray = None) -> List[int]:
        """
        Detect changepoints using a penalty parameter.
        
        The penalty controls the trade-off between model fit and number of changepoints:
        - Low penalty (e.g., 0.5-2): Many changepoints, sensitive to small changes
        - Medium penalty (e.g., 3-10): Moderate number of significant changepoints
        - High penalty (e.g., 20+): Few changepoints, only major regime shifts
        
        Args:
            pen (float): Penalty value (higher = fewer changepoints)
            signal (np.ndarray): Signal to use (if None, uses fitted signal)
            
        Returns:
            List[int]: Indices of detected changepoints
        """
        if self.algo is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Detecting changepoints with penalty={pen}")
        
        # Get signal length from stored signal
        if signal is None:
            if self.signal is None:
                raise ValueError("No signal available. Pass signal or fit() first.")
            signal_length = len(self.signal)
        else:
            signal_length = len(signal)
        
        # Predict changepoints
        breakpoints = self.algo.predict(pen=pen)
        
        # ruptures returns the end of each segment, including the last index
        # We want the actual breakpoints (excluding the last index)
        self.breakpoints = [bp for bp in breakpoints if bp < signal_length]
        
        logger.info(f"✓ Detected {len(self.breakpoints)} changepoints")
        
        return self.breakpoints
    
    def predict_with_n_breaks(self, n_bkps: int, signal: np.ndarray = None) -> List[int]:
        """
        Detect a fixed number of changepoints.
        
        Useful for:
        - Initial testing to verify the model works
        - Finding the top N most significant regime changes
        
        Args:
            n_bkps (int): Number of changepoints to detect
            signal (np.ndarray): Signal to use (if None, uses fitted signal)
            
        Returns:
            List[int]: Indices of detected changepoints
        """
        if self.algo is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Detecting top {n_bkps} changepoints")
        
        # Get signal length from stored signal
        if signal is None:
            if self.signal is None:
                raise ValueError("No signal available. Pass signal or fit() first.")
            signal_length = len(self.signal)
        else:
            signal_length = len(signal)
        
        # Use a low penalty and then select top N breaks
        # This is a workaround since Pelt doesn't support n_bkps directly
        try:
            breakpoints = self.algo.predict(n_bkps=n_bkps)
        except TypeError:
            # If n_bkps not supported, use adaptive penalty search
            logger.info("n_bkps not supported, using adaptive penalty search...")
            
            # Binary search for the right penalty
            for pen in [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20]:
                breakpoints = self.algo.predict(pen=pen)
                valid_breaks = [bp for bp in breakpoints if bp < signal_length]
                
                if len(valid_breaks) <= n_bkps:
                    logger.info(f"  Found {len(valid_breaks)} breaks with penalty={pen}")
                    break
            
            # If we still have too many, take the first N
            if len(valid_breaks) > n_bkps:
                breakpoints = valid_breaks[:n_bkps]
        
        # Exclude the last index
        self.breakpoints = [bp for bp in breakpoints if bp < signal_length]
        
        logger.info(f"✓ Detected {len(self.breakpoints)} changepoints")
        
        return self.breakpoints
    
    def map_breakpoints_to_dates(self, dates: np.ndarray) -> pd.DatetimeIndex:
        """
        Convert breakpoint indices to actual dates.
        
        Args:
            dates (np.ndarray): Array of dates corresponding to the signal
            
        Returns:
            pd.DatetimeIndex: Dates of changepoints
        """
        if self.breakpoints is None:
            raise ValueError("No breakpoints detected. Call predict_with_penalty() or predict_with_n_breaks() first.")
        
        if len(dates) < max(self.breakpoints):
            raise ValueError("Dates array is shorter than breakpoint indices")
        
        # Map indices to dates
        self.breakpoint_dates = pd.DatetimeIndex([dates[bp] for bp in self.breakpoints])
        
        logger.info(f"Mapped {len(self.breakpoint_dates)} breakpoints to dates")
        for i, date in enumerate(self.breakpoint_dates, 1):
            logger.info(f"  Changepoint {i}: {date.date()}")
        
        return self.breakpoint_dates
    
    def tune_penalty(self, 
                     signal: np.ndarray, 
                     penalty_range: List[float] = None,
                     expected_range: Tuple[int, int] = (5, 20)) -> pd.DataFrame:
        """
        Tune the penalty parameter by testing multiple values.
        
        Helps find the optimal penalty that detects a reasonable number of changepoints.
        
        Args:
            signal (np.ndarray): Input signal
            penalty_range (List[float]): List of penalty values to test
            expected_range (Tuple[int, int]): Expected range of changepoints (min, max)
            
        Returns:
            pd.DataFrame: Results showing penalty vs number of changepoints
        """
        if penalty_range is None:
            # Default range: exponentially spaced penalties
            penalty_range = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
        
        logger.info(f"Tuning penalty parameter across {len(penalty_range)} values")
        
        results = []
        
        # Fit the model once
        self.fit(signal)
        
        for pen in penalty_range:
            try:
                breakpoints = self.predict_with_penalty(pen, signal=signal)
                n_breaks = len(breakpoints)
                
                # Check if in expected range
                in_range = expected_range[0] <= n_breaks <= expected_range[1]
                
                results.append({
                    'penalty': pen,
                    'n_changepoints': n_breaks,
                    'in_expected_range': in_range
                })
                
                logger.info(f"  Penalty {pen:6.1f} → {n_breaks:3d} changepoints {'✓' if in_range else ''}")
                
            except Exception as e:
                logger.warning(f"  Penalty {pen:6.1f} → Error: {e}")
                results.append({
                    'penalty': pen,
                    'n_changepoints': None,
                    'in_expected_range': False
                })
        
        results_df = pd.DataFrame(results)
        
        # Find recommended penalty
        valid_results = results_df[results_df['in_expected_range'] == True]
        if len(valid_results) > 0:
            # Choose the highest penalty that still gives a good number of breaks
            recommended = valid_results.iloc[-1]
            logger.info(f"\n✓ Recommended penalty: {recommended['penalty']} "
                       f"({recommended['n_changepoints']} changepoints)")
        
        return results_df
    
    def get_segment_statistics(self, signal: np.ndarray = None) -> pd.DataFrame:
        """
        Calculate statistics for each segment between changepoints.
        
        Args:
            signal (np.ndarray): Input signal (if None, uses stored signal)
            
        Returns:
            pd.DataFrame: Statistics for each segment
        """
        if self.breakpoints is None:
            raise ValueError("No breakpoints detected.")
        
        # Use stored signal if not provided
        if signal is None:
            if self.signal is None:
                raise ValueError("No signal available. Pass signal or fit() first.")
            signal = self.signal
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        # Create segments
        segments = []
        start_idx = 0
        
        for bp in self.breakpoints:
            segment = signal[start_idx:bp]
            segments.append({
                'start_idx': start_idx,
                'end_idx': bp,
                'length': len(segment),
                'mean': np.mean(segment),
                'std': np.std(segment),
                'min': np.min(segment),
                'max': np.max(segment)
            })
            start_idx = bp
        
        stats_df = pd.DataFrame(segments)
        
        return stats_df


if __name__ == "__main__":
    # Test the module
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    loader = DataLoader("../../Nifty_50.csv")
    data = loader.load_data()
    data = loader.clean_data()
    
    engineer = FeatureEngineer(data)
    engineer.calculate_log_returns()
    engineer.calculate_rolling_volatility(window=30)
    
    signal, dates = engineer.get_clean_signal('rolling_vol_30d')
    
    detector = ChangepointDetector(model="l2")
    detector.fit(signal)
    
    # Test with fixed number of breaks
    breakpoints = detector.predict_with_n_breaks(n_bkps=10)
    breakpoint_dates = detector.map_breakpoints_to_dates(dates)
    
    print(f"\nTop 10 changepoints detected:")
    for date in breakpoint_dates:
        print(f"  {date.date()}")
